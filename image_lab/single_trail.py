"""单车灯光轨跟踪合成。

固定机位短序列（第一张为空场参考帧），从用户在第二张上指出的亮斑开始，
沿用光轨流程的亮度门槛检出分离亮斑（排除遮罩内像素不计），只跟踪一条光轨：
相邻照片按亮斑中心距离限制转移，最多跨过一张漏检照片（按两张总位移判断），
在“保留实拍亮斑最多、数量相同则总位移更短”的目标下选路；
合成图以参考背景提亮为底，只叠加被选中的亮斑。
"""
import numpy as np
from scipy import ndimage as ndi
from . import validation as v
from .arrays import luminance, srgb_decode, srgb_encode

MIN_FRAMES = 4
MAX_FRAMES = 6
MAX_LONG_EDGE = 256
MAX_BLOBS_PER_FRAME = 12
START_FRAME_INDEX = 1  # 用户在第二张（索引1）上指出起始亮斑
MAX_GAP = 1  # 中间最多允许漏掉一张

LIMITS = {
    'frames_min': MIN_FRAMES,
    'frames_max': MAX_FRAMES,
    'long_edge': MAX_LONG_EDGE,
    'blobs_per_frame': MAX_BLOBS_PER_FRAME,
    'max_gap': MAX_GAP,
    'start_frame': START_FRAME_INDEX + 1,
}


def settings(body):
    body = v.object_value(body, '单车灯跟踪设置')
    result = {
        'threshold': v.number(body.get('threshold', .05), '轨迹亮度阈值', 0, .8),
        'max_shift': v.integer(body.get('max_shift', 40), '单张最大位移像素', 1, MAX_LONG_EDGE),
        'start_point': None,
    }
    start = body.get('start_point')
    if start is not None:
        point = v.vector(start, '起始亮斑位置', 2, 0, MAX_LONG_EDGE - 1)
        result['start_point'] = [float(point[0]), float(point[1])]
    return result


def _check_scope(prepared):
    frame_count = len(prepared['frames'])
    if not MIN_FRAMES <= frame_count <= MAX_FRAMES:
        raise v.Invalid(f'单车灯跟踪一次只处理{MIN_FRAMES}至{MAX_FRAMES}张照片，当前启用{frame_count}张')
    width, height = prepared['size']
    if max(width, height) > MAX_LONG_EDGE:
        raise v.Invalid(f'单车灯跟踪工作画布最长边不得超过{MAX_LONG_EDGE}像素，请在工程设置中调整工作长边')
    if prepared['reference_index'] != 0:
        raise v.Invalid('单车灯跟踪要求第一张照片（空场）作为参考帧')


def detect_series(prepared, config):
    """逐帧检出亮斑，返回每帧的标签图与亮斑中心/像素统计。"""
    _check_scope(prepared)
    stack = [srgb_decode(image) for image in prepared['images']]
    reference_lum = luminance(stack[prepared['reference_index']])
    series = []
    for index, image in enumerate(stack):
        valid = np.asarray(prepared['validities'][index], dtype=np.float32) > .5
        bright = luminance(image) - reference_lum
        eligible = (bright > config['threshold']) & valid
        labels, count = ndi.label(eligible)
        if count > MAX_BLOBS_PER_FRAME:
            raise v.Invalid(
                f'第{index + 1}张照片检出{count}块亮斑，超过每张{MAX_BLOBS_PER_FRAME}块的上限，请提高亮度阈值或扩大排除遮罩'
            )
        blobs = []
        if count:
            records = ndi.find_objects(labels)
            for blob_id, region in enumerate(records, start=1):
                if region is None:
                    continue
                area = int((labels[region] == blob_id).sum())
                ys, xs = np.nonzero(labels == blob_id)
                blobs.append({
                    'id': int(blob_id),
                    'center': [float(xs.mean()), float(ys.mean())],
                    'area': area,
                    'bounds': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                })
        series.append({'frame_index': index, 'labels': labels.astype(np.int16), 'blobs': blobs})
    return stack, series


def _blob_payload(entry, frame):
    return [
        {
            'id': blob['id'],
            'center': [round(blob['center'][0], 3), round(blob['center'][1], 3)],
            'area': blob['area'],
            'bounds': blob['bounds'],
            'frame_index': frame,
        }
        for blob in entry['blobs']
    ]


def series_payload(prepared, config):
    _, series = detect_series(prepared, config)
    return {
        'limits': dict(LIMITS),
        'start_frame': START_FRAME_INDEX + 1,
        'reference_frame': prepared['reference_index'] + 1,
        'frames': [
            {
                'frame_index': entry['frame_index'],
                'frame_id': prepared['frames'][entry['frame_index']]['id'],
                'name': prepared['frames'][entry['frame_index']]['name'],
                'blob_count': len(entry['blobs']),
                'blobs': _blob_payload(entry, entry['frame_index']),
            }
            for entry in series
        ],
    }


def _resolve_start(series, point):
    entry = series[START_FRAME_INDEX]
    x, y = point
    px, py = int(round(x)), int(round(y))
    labels = entry['labels']
    h, w = labels.shape
    if 0 <= px < w and 0 <= py < h and labels[py, px] > 0:
        blob_id = int(labels[py, px])
        return next(blob for blob in entry['blobs'] if blob['id'] == blob_id)
    if not entry['blobs']:
        raise v.Invalid('起始照片上没有可用亮斑，请降低亮度阈值或检查排除遮罩')
    nearest = min(entry['blobs'], key=lambda blob: (blob['center'][0] - x) ** 2 + (blob['center'][1] - y) ** 2)
    return nearest


def _distance(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def track_path(series, start_blob, config):
    """动态规划选路：亮斑数最多优先，其次中心总位移最短。

    状态为“当前帧匹配到的亮斑”；转移可走到下一帧（位移不超过单张上限），
    也可跨过一张漏检帧（总位移不超过两张上限，缺帧不补灯）。
    """
    max_shift = config['max_shift']
    last = len(series) - 1
    # 每帧每个亮斑的最优状态：(已保留亮斑数, 总位移, 前驱帧, 前驱亮斑id)
    states = {START_FRAME_INDEX: {start_blob['id']: (1, 0.0, None, None)}}
    for frame in range(START_FRAME_INDEX, last):
        current = states.get(frame, {})
        if not current:
            continue
        candidates = series[frame]['blobs']
        for gap in range(1, MAX_GAP + 2):  # gap=1 相邻帧；gap=2 跨过一张
            target = frame + gap
            if target > last:
                continue
            limit = max_shift * gap
            states.setdefault(target, {})
            for blob_id, (count, total, _, _) in current.items():
                center = next(blob['center'] for blob in candidates if blob['id'] == blob_id)
                for candidate in series[target]['blobs']:
                    shift = _distance(center, candidate['center'])
                    if shift > limit:
                        continue
                    trial = (count + 1, total + shift, frame, blob_id)
                    previous = states[target].get(candidate['id'])
                    if previous is None or (trial[0], -trial[1]) > (previous[0], -previous[1]):
                        states[target][candidate['id']] = trial
    # 在所有可达状态中选：亮斑数最多，总位移最短；再平手取结束帧更晚、亮斑id更小
    best_key = None
    best_state = None
    for frame in sorted(states):
        for blob_id, state in states[frame].items():
            key = (state[0], -state[1], frame, -blob_id)
            if best_key is None or key > best_key:
                best_key = key
                best_state = state
                best_frame, best_blob = frame, blob_id
    # 回溯
    chain = []
    frame, blob_id, state = best_frame, best_blob, best_state
    while frame is not None:
        chain.append((frame, blob_id))
        _, _, prev_frame, prev_blob = state
        if prev_frame is None:
            break
        frame, blob_id = prev_frame, prev_blob
        state = states[frame][blob_id]
    chain.reverse()
    selected = {}
    links = []
    previous = None
    total_shift = 0.0
    for frame, blob_id in chain:
        blob = next(item for item in series[frame]['blobs'] if item['id'] == blob_id)
        selected[frame] = blob
        if previous is not None:
            prev_frame, prev_blob = previous
            skipped = frame - prev_frame - 1
            shift = _distance(prev_blob['center'], blob['center'])
            total_shift += shift
            links.append({
                'from_frame': prev_frame,
                'to_frame': frame,
                'distance': round(shift, 3),
                'limit': max_shift * (frame - prev_frame),
                'skipped_frames': [prev_frame + 1 + offset for offset in range(skipped)],
            })
        previous = (frame, blob)
    return selected, links, round(total_shift, 3)


def combine(prepared, config):
    if config.get('start_point') is None:
        raise v.Invalid('请先在第二张照片上点选起始亮斑')
    stack, series = detect_series(prepared, config)
    start_blob = _resolve_start(series, config['start_point'])
    selected, links, total_shift = track_path(series, start_blob, config)

    reference = stack[prepared['reference_index']].copy()
    result = reference.copy()
    coverage = np.zeros(result.shape[:2], dtype=np.float32)
    selected_frames, skipped_frames = [], []
    for frame, blob in selected.items():
        component = series[frame]['labels'] == blob['id']
        target = np.maximum(result, stack[frame])
        result = np.where(component[:, :, None], target, result)
        coverage[component] = 1
        selected_frames.append(frame)
    for frame in range(START_FRAME_INDEX, len(series)):
        if frame not in selected:
            skipped_frames.append(frame)

    image = np.clip(srgb_encode(result), 0, 1)
    validity = np.stack(prepared['validities']).max(axis=0)
    image[validity < .001] = 0
    fields = {
        'image': image.astype(np.float32),
        'valid': np.clip(validity, 0, 1).astype(np.float32),
        'trail_mask': coverage.astype(np.float32),
    }
    frame_lookup = prepared['frames']
    report = {
        'mode': 'single_trail',
        'note': '从第二张点选亮斑起只跟踪一条光轨；漏检帧不补灯，合成沿用参考背景提亮并只叠加选中亮斑。',
        'limits': dict(LIMITS),
        'threshold': config['threshold'],
        'max_shift': config['max_shift'],
        'start_point': [round(config['start_point'][0], 3), round(config['start_point'][1], 3)],
        'start_blob': {'frame_index': START_FRAME_INDEX, 'id': start_blob['id'],
                       'center': [round(start_blob['center'][0], 3), round(start_blob['center'][1], 3)]},
        'selected_frames': [
            {
                'frame_index': frame,
                'frame_id': frame_lookup[frame]['id'],
                'name': frame_lookup[frame]['name'],
                'blob_id': selected[frame]['id'],
                'center': [round(selected[frame]['center'][0], 3), round(selected[frame]['center'][1], 3)],
                'area': selected[frame]['area'],
            }
            for frame in selected_frames
        ],
        'skipped_frames': [
            {'frame_index': frame, 'frame_id': frame_lookup[frame]['id'], 'name': frame_lookup[frame]['name']}
            for frame in skipped_frames
        ],
        'links': links,
        'selected_count': len(selected_frames),
        'total_shift': total_shift,
    }
    return fields, report


def track_preview(prepared, config):
    """供交互预览：完整报告（含各帧亮斑）与合成PNG字节。"""
    from .codecs import png_bytes
    fields, report = combine(prepared, config)
    report['detection'] = series_payload(prepared, config)['frames']
    return png_bytes(fields['image']), report
