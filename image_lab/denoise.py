import numpy as np
from .arrays import srgb_decode, srgb_encode, normalize_weights, weighted_rgb, safe_divide
from .validation import Invalid


def sigma_clip(stack, valid, sigma=2, iterations=2):
    accepted = valid[:, :, :, None] > .5
    accepted = np.broadcast_to(accepted, stack.shape).copy()
    for _ in range(iterations):
        count = accepted.sum(axis=0)
        mean = safe_divide(np.sum(stack * accepted, axis=0), count)
        variance = safe_divide(np.sum((stack - mean) ** 2 * accepted, axis=0), count)
        deviation = np.sqrt(variance)
        reject = np.abs(stack - mean) > sigma * np.maximum(deviation, 1e-5)
        candidate = accepted & ~reject
        empty = candidate.sum(axis=0) == 0
        candidate[:, empty] = accepted[:, empty]
        accepted = candidate
    mean = safe_divide(np.sum(stack * accepted, axis=0), accepted.sum(axis=0))
    rejected = 1 - accepted.mean(axis=(0, 3))
    return mean, rejected.astype(np.float32), accepted.mean(axis=3).astype(np.float32)


def _compatible_pairs(colors, valid, limit):
    """逐像素返回帧对兼容矩阵：两张照片在该处均有效，且RGB三个通道差都不超过限值。"""
    diff = np.abs(colors[:, None, :, :] - colors[None, :, :, :])
    close = np.all(diff <= limit, axis=-1)
    both_valid = (valid[:, None, :, :] > .5) & (valid[None, :, :, :] > .5)
    return close & both_valid


def color_consensus(srgb_stack, linear_stack, valid, limit, reference_index):
    """每个像素在有效照片中选互相全通道接近且张数最多的一组，等权平均。

    颜色接近程度按8位RGB级数解释（限值已换算为0-1浮点差），一张照片的红绿蓝
    整体取或整体舍；等张数时优先平均颜色更接近参考照片的组，参考无效则按原照片
    顺序取最先达到该张数的组。
    """
    frame_count, height, width, _ = srgb_stack.shape
    pairs = _compatible_pairs(srgb_stack, valid, limit)
    # 逐像素记录当前最优组：成员掩码、张数、组平均色到参考色的距离、最小编号锚点
    best_members = np.zeros((frame_count, height, width), dtype=bool)
    best_count = np.zeros((height, width), np.float32)
    best_distance = np.full((height, width), np.inf, np.float32)
    ref_valid = valid[reference_index] > .5
    ref_colors = srgb_stack[reference_index]
    for anchor in range(frame_count):
        # 以第anchor张为最小编号成员的候选组，天然实现“按原照片顺序”的兜底规则
        members = np.zeros((frame_count, height, width), dtype=bool)
        members[anchor] = valid[anchor] > .5
        for other in range(anchor + 1, frame_count):
            members[other] = pairs[anchor, other]
        count = members.sum(axis=0).astype(np.float32)
        candidate_mean = safe_divide(
            np.sum(srgb_stack * members[:, :, :, None], axis=0),
            np.maximum(count[:, :, None], 1))
        distance = np.sqrt(np.sum((candidate_mean - ref_colors) ** 2, axis=2)).astype(np.float32)
        larger = count > best_count
        # 参考有效：等张数比组平均色到参考的距离；参考无效：保留最先达到该张数的组
        tied_closer = (count == best_count) & (count > 0) & ref_valid & (distance < best_distance - 1e-7)
        update = larger | tied_closer
        if not np.any(update):
            continue
        best_members = np.where(update[None, :, :], members, best_members)
        best_count = np.where(update, count, best_count)
        best_distance = np.where(update & ref_valid, distance, np.where(update, np.inf, best_distance))
    weights = best_members.astype(np.float32)
    srgb_mean = safe_divide(
        np.sum(srgb_stack * weights[:, :, :, None], axis=0),
        np.maximum(best_count[:, :, None], 1))
    linear_mean = safe_divide(
        np.sum(linear_stack * weights[:, :, :, None], axis=0),
        np.maximum(best_count[:, :, None], 1))
    return srgb_mean, linear_mean, weights, best_count


def check_color_consensus_limits(prepared):
    """整组颜色一致选项的首期规模护栏：最多5张照片、工作画布长边不超过384像素。"""
    if len(prepared['frames']) > 5:
        raise Invalid('整组颜色一致降噪首期最多支持5张启用照片，当前为' + str(len(prepared['frames'])) + '张')
    if max(prepared['size']) > 384:
        raise Invalid('整组颜色一致降噪首期要求工作画布长边不超过384像素，当前长边为' + str(max(prepared['size'])) + '像素')


def denoise_stack(prepared, settings):
    source = prepared['images']
    linear_images = [srgb_decode(image) for image in source]
    use_linear = settings['linear']
    images = linear_images if use_linear else source
    stack = np.stack(images)
    valid = np.stack(prepared['validities'])
    weights = valid * np.asarray([f['weight'] for f in prepared['frames']])[:, None, None]
    rejected = np.zeros_like(valid[0])
    method = settings['method']
    sample_count = None
    if method == 'sigma':
        image, rejected, accepted = sigma_clip(stack, weights, settings['sigma'], settings['iterations'])
        weights = accepted * weights
    elif method == 'median':
        masked = np.ma.array(stack, mask=np.broadcast_to((valid < .5)[:, :, :, None], stack.shape))
        image = np.ma.median(masked, axis=0).filled(0).astype(np.float32)
    elif method == 'trimmed':
        masked = np.where(valid[:, :, :, None] > .5, stack, np.nan)
        lo = np.nanquantile(masked, settings['trim'], axis=0)
        hi = np.nanquantile(masked, 1 - settings['trim'], axis=0)
        accepted = (stack >= lo) & (stack <= hi) & (valid[:, :, :, None] > .5)
        image = safe_divide((stack * accepted).sum(axis=0), accepted.sum(axis=0))
        rejected = 1 - accepted.mean(axis=(0, 3))
        weights = accepted.mean(axis=3) * weights
    elif method == 'color_consensus':
        check_color_consensus_limits(prepared)
        limit = settings['color_tolerance'] / 255.
        srgb_mean, linear_mean, consensus_weights, sample_count = color_consensus(
            np.stack(source), np.stack(linear_images), valid, limit, prepared['reference_index'])
        image = linear_mean if use_linear else srgb_mean
        weights = consensus_weights
    else:
        if method == 'mean':
            weights = valid
        image = weighted_rgb(images, normalize_weights(weights))
    variance = np.sum((stack - image[None]) ** 2 * valid[:, :, :, None], axis=0)
    variance = safe_divide(variance, valid.sum(axis=0)[:, :, None])
    if use_linear:
        image = srgb_encode(image)
    coverage = valid.max(axis=0)
    image[coverage < .001] = 0
    effective = safe_divide(np.sum(weights, axis=0) ** 2, np.sum(weights ** 2, axis=0))
    fields = {'image': np.clip(image, 0, 1), 'valid': coverage, 'variance': variance.mean(axis=2), 'rejection': rejected, 'effective_frames': effective / len(images)}
    if sample_count is not None:
        fields['sample_count'] = sample_count
    report = {'mode': 'denoise', 'method': method, 'linear_space': use_linear, 'mean_effective_frames': float(effective.mean()), 'ideal_noise_reduction_factor': float(np.sqrt(np.maximum(effective.mean(), 1))), 'mean_rejected': float(rejected.mean()), 'note': '理论降噪因子假设独立同分布噪声，配准误差、压缩与运动会降低实际收益。'}
    if method == 'color_consensus':
        report['color_tolerance_8bit'] = settings['color_tolerance']
        report['mean_sample_count'] = float(sample_count[coverage > .001].mean()) if np.any(coverage > .001) else 0.
        report['max_sample_count'] = int(sample_count.max())
        report['note'] = '每个位置只在配准后有效且未被遮罩排除的照片中，整组取舍RGB并选择互相全通道接近且张数最多的一组；8位级数容差' + str(settings['color_tolerance']) + '。'
    return fields, report
