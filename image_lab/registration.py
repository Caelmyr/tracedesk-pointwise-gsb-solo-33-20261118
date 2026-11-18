import time
import cv2
import numpy as np
from scipy.optimize import linprog
from .arrays import luminance, normalize
from .geometry import warp, normalized_matrix, overlap_metrics
from . import validation as v


def grayscale(image):
    return luminance(image).astype(np.float32)


def feature_gray(image):
    gray = normalize(grayscale(image), 1, 99)
    raster = np.rint(gray * 255).astype(np.uint8)
    return cv2.createCLAHE(2., (8, 8)).apply(raster)


def phase_align(reference, moving):
    a, b = grayscale(reference), grayscale(moving)
    a = np.log1p(a * 20)
    b = np.log1p(b * 20)
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(a, b, window)
    if not np.isfinite(shift).all():
        raise v.Invalid('相位相关无法估计位移')
    matrix = np.array([[1, 0, -shift[0]], [0, 1, -shift[1]], [0, 0, 1]], np.float64)
    return matrix, {'response': float(response), 'shift': list(shift), 'method': 'phase'}


def orb_align(reference, moving, settings):
    detector = cv2.ORB_create(nfeatures=settings['features'], fastThreshold=10)
    a, des_a = detector.detectAndCompute(feature_gray(reference), None)
    b, des_b = detector.detectAndCompute(feature_gray(moving), None)
    if des_a is None or des_b is None or len(a) < 8 or len(b) < 8:
        raise v.Invalid('特征不足，建议使用相位相关或手动变换')
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    candidates = matcher.knnMatch(des_b, des_a, k=2)
    matches = [first for pair in candidates if len(pair) == 2 for first, second in [pair] if first.distance < settings['ratio'] * second.distance]
    if len(matches) < 6:
        raise v.Invalid('可靠匹配不足6对')
    source = np.float32([b[m.queryIdx].pt for m in matches])
    target = np.float32([a[m.trainIdx].pt for m in matches])
    if settings['motion'] == 'homography':
        matrix, inliers = cv2.findHomography(source, target, cv2.RANSAC, settings['ransac'])
    elif settings['motion'] == 'translation':
        differences = target - source
        shift = np.median(differences, axis=0)
        inliers = (np.linalg.norm(differences - shift, axis=1) < settings['ransac']).astype(np.uint8)
        matrix = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]], np.float64)
    else:
        estimator = cv2.estimateAffinePartial2D if settings['motion'] == 'euclidean' else cv2.estimateAffine2D
        affine, inliers = estimator(source, target, method=cv2.RANSAC, ransacReprojThreshold=settings['ransac'])
        matrix = np.vstack([affine, [0, 0, 1]]) if affine is not None else None
    if matrix is None or inliers is None or np.sum(inliers) < 5:
        raise v.Invalid('RANSAC无法获得稳定变换')
    projected = cv2.perspectiveTransform(source[None], matrix)[0]
    error = np.linalg.norm(projected - target, axis=1)
    selected = inliers.ravel() > 0
    lines = [{'source': source[i].tolist(), 'target': target[i].tolist(), 'inlier': bool(selected[i])} for i in np.linspace(0, len(matches) - 1, min(80, len(matches))).astype(int)]
    return matrix, {'method': 'orb', 'keypoints': [len(a), len(b)], 'matches': len(matches), 'inliers': int(selected.sum()), 'median_reprojection_error': float(np.median(error[selected])), 'match_lines': lines}


def ecc_align(reference, moving, settings):
    mapping = {'translation': cv2.MOTION_TRANSLATION, 'euclidean': cv2.MOTION_EUCLIDEAN, 'affine': cv2.MOTION_AFFINE, 'homography': cv2.MOTION_HOMOGRAPHY}
    motion = mapping[settings['motion']]
    seed, _ = phase_align(reference, moving)
    inverse = np.linalg.inv(seed).astype(np.float32)
    matrix = inverse if motion == cv2.MOTION_HOMOGRAPHY else inverse[:2]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, settings['iterations'], settings['epsilon'])
    try:
        score, result = cv2.findTransformECC(normalize(grayscale(reference), 1, 99), normalize(grayscale(moving), 1, 99), matrix, motion, criteria, None, 5)
    except cv2.error as error:
        raise v.Invalid('ECC未收敛，请先尝试相位相关或特征配准') from error
    if result.shape == (2, 3):
        result = np.vstack([result, [0, 0, 1]])
    return np.linalg.inv(result), {'method': 'ecc', 'correlation': float(score)}


def mtb_align(reference, moving):
    a = normalize(grayscale(reference), 1, 99)
    b = normalize(grayscale(moving), 1, 99)
    median_a, median_b = np.median(a), np.median(b)
    bitmap_a = (a > median_a).astype(np.float32)
    bitmap_b = (b > median_b).astype(np.float32)
    exclusion_a = (np.abs(a - median_a) > 4 / 255).astype(np.float32)
    exclusion_b = (np.abs(b - median_b) > 4 / 255).astype(np.float32)
    prior, prior_details = phase_align(reference, moving)
    seed_x, seed_y = int(round(prior[0, 2])), int(round(prior[1, 2]))
    h, w = a.shape
    best = None
    for dy in range(seed_y - 3, seed_y + 4):
        for dx in range(seed_x - 3, seed_x + 4):
            matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)
            shifted = warp(bitmap_b, matrix, (w, h), True)
            exclusion = warp(exclusion_b, matrix, (w, h), True) * exclusion_a
            count = float(exclusion.sum())
            if count < h * w * .1:
                continue
            mismatch = float(np.sum(np.abs(bitmap_a - shifted) * exclusion) / count)
            tie_break = (dx - prior[0, 2]) ** 2 + (dy - prior[1, 2]) ** 2
            candidate = (mismatch, tie_break, dx, dy)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise v.Invalid('中值阈值位图缺少有效纹理')
    error, _, dx, dy = best
    matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)
    return matrix, {'method': 'mtb', 'shift': [dx, dy], 'bitmap_mismatch': error, 'phase_prior_response': prior_details['response'], 'note': '相位初值附近搜索中值阈值位图异或误差，排除中灰不稳定像素。'}


def alignment_quality(reference, moving, matrix):
    h, w = reference.shape[:2]
    aligned = warp(moving, matrix, (w, h))
    support = warp(np.ones((h, w), np.float32), matrix, (w, h)) > .99
    if support.mean() < .01:
        return {'overlap': float(support.mean()), 'correlation': 0., 'normalized_mae': 1.}
    a = grayscale(reference)[support]
    b = grayscale(aligned)[support]
    a = (a - a.mean()) / max(a.std(), 1e-6)
    b = (b - b.mean()) / max(b.std(), 1e-6)
    return {'overlap': float(support.mean()), 'correlation': float(np.mean(a * b)), 'normalized_mae': float(np.mean(np.abs(a - b)))}


def _translation(matrix):
    linear = matrix[:2, :2]
    if not np.isfinite(matrix).all() or np.abs(linear - np.eye(2)).max() > 1e-4 or abs(np.linalg.det(matrix)) < 1e-8:
        raise v.Invalid('串接配准只接受纯平移结果')
    return float(matrix[0, 2]), float(matrix[1, 2])


def _pair_translation(anchor, moving, settings):
    forced = dict(settings)
    forced['motion'] = 'translation'
    if forced['method'] == 'phase':
        matrix, details = phase_align(anchor, moving)
    elif forced['method'] == 'orb':
        matrix, details = orb_align(anchor, moving, forced)
    elif forced['method'] == 'ecc':
        matrix, details = ecc_align(anchor, moving, forced)
    else:
        matrix, details = mtb_align(anchor, moving)
    dx, dy = _translation(matrix)
    return dx, dy, details


def _trusted_edges(pairs, count, reference_index):
    adjacency = {index: [] for index in range(count)}
    for edge in pairs:
        if edge['trusted']:
            adjacency[edge['a']].append(edge['b'])
            adjacency[edge['b']].append(edge['a'])
    reachable = {reference_index}
    queue = [reference_index]
    while queue:
        current = queue.pop()
        for neighbor in adjacency[current]:
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)
    return reachable


def _minimax_positions(pairs, count, reference_index, reachable):
    nodes = [index for index in range(count) if index != reference_index and index in reachable]
    # 变量排布：先所有节点的 x，再所有节点的 y，最后是最大绝对偏差 t
    x_column = {index: pos for pos, index in enumerate(nodes)}
    y_column = {index: len(nodes) + pos for pos, index in enumerate(nodes)}
    t_index = 2 * len(nodes)
    objective = np.zeros(t_index + 1)
    objective[t_index] = 1.
    upper, rhs = [], []
    edges = [edge for edge in pairs if edge['trusted'] and edge['a'] in reachable and edge['b'] in reachable]
    positions = {reference_index: np.zeros(2)}
    residuals = {pair_index: None for pair_index in range(len(pairs))}
    if not edges:
        return positions, residuals, edges
    for edge in edges:
        for axis, observed in enumerate(edge['shift']):
            signs = np.zeros(t_index + 1)
            columns = x_column if axis == 0 else y_column
            # 配对位移约定：p_b - p_a = observed，残差 = (p_b - p_a) - observed
            if edge['a'] != reference_index:
                signs[columns[edge['a']]] = -1.
            if edge['b'] != reference_index:
                signs[columns[edge['b']]] = 1.
            # -t <= (p_b - p_a - observed) <= t
            signs[t_index] = -1.
            upper.append(signs.copy())
            rhs.append(observed)
            reverse = -signs
            reverse[t_index] = -1.
            upper.append(reverse)
            rhs.append(-observed)
    result = linprog(objective, A_ub=np.array(upper), b_ub=np.array(rhs), bounds=[(None, None)] * t_index + [(0., None)], method='highs')
    if not result.success:
        raise v.Invalid('串接位移全局求解失败：' + str(result.message))
    for index in nodes:
        positions[index] = np.array([result.x[x_column[index]], result.x[y_column[index]]])
    residuals = {}
    for pair_index, edge in enumerate(pairs):
        if not edge['trusted'] or edge['a'] not in reachable or edge['b'] not in reachable:
            residuals[pair_index] = None
            continue
        residual = (positions[edge['b']] - positions[edge['a']]) - np.asarray(edge['shift'], dtype=np.float64)
        residuals[pair_index] = {'x': float(residual[0]), 'y': float(residual[1]), 'max_abs': float(np.abs(residual).max())}
    return positions, residuals, edges


def chain_align_set(prepared, settings, progress=None, cancelled=None):
    images = prepared['images']
    count = len(images)
    reference_index = prepared['reference_index']
    reference = images[reference_index]
    width, height = prepared['size']
    if not 3 <= count <= 5:
        raise v.Invalid('串接配准仅支持3到5张工作照片')
    if any(image.shape != reference.shape for image in images):
        raise v.Invalid('串接配准要求所有照片尺寸相同')
    if max(reference.shape[:2]) > 384:
        raise v.Invalid('串接配准仅支持最长边不超过384的工作图')
    frames = prepared['frames']
    pairs = []
    total = count * (count - 1) // 2
    done = 0
    for a in range(count):
        for b in range(a + 1, count):
            if cancelled and cancelled():
                raise InterruptedError('配准任务已取消')
            started = time.monotonic()
            record = {'a': a, 'b': b, 'a_frame': frames[a]['id'], 'b_frame': frames[b]['id'], 'a_name': frames[a]['name'], 'b_name': frames[b]['name'], 'shift': None, 'overlap': None, 'correlation': None, 'normalized_mae': None, 'trusted': False, 'method': settings['method'], 'reason': None, 'seconds': 0.}
            try:
                dx, dy, details = _pair_translation(images[a], images[b], settings)
                matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)
                quality = alignment_quality(images[a], images[b], matrix)
                record.update(shift=[dx, dy], overlap=quality['overlap'], correlation=quality['correlation'], normalized_mae=quality['normalized_mae'], seconds=time.monotonic() - started)
                record['method'] = details.get('method', settings['method'])
                if quality['overlap'] < settings['minimum_overlap']:
                    record['reason'] = '重叠比例低于下限'
                elif quality['correlation'] < settings['minimum_correlation']:
                    record['reason'] = '相关性低于下限'
                else:
                    record['trusted'] = True
            except (v.Invalid, cv2.error) as error:
                record['reason'] = '单对配准失败：' + str(error)
            pairs.append(record)
            done += 1
            if progress:
                progress(.35 + .45 * done / max(total, 1), '串接配准 ' + frames[a]['name'] + ' / ' + frames[b]['name'])
    reachable = _trusted_edges(pairs, count, reference_index)
    positions, residuals, used_edges = _minimax_positions(pairs, count, reference_index, reachable)
    for pair_index, residual in residuals.items():
        pairs[pair_index]['residual'] = residual
    used_pair_indices = {index: [] for index in range(count)}
    for pair_index, edge in enumerate(pairs):
        if edge['trusted'] and edge['a'] in reachable and edge['b'] in reachable:
            used_pair_indices[edge['a']].append(pair_index)
            used_pair_indices[edge['b']].append(pair_index)
    result = []
    for index, frame in enumerate(frames):
        if cancelled and cancelled():
            raise InterruptedError('配准任务已取消')
        started = time.monotonic()
        if index == reference_index:
            matrix = np.eye(3)
            quality = alignment_quality(reference, reference, matrix)
            details = {'method': 'reference', 'connected': True, 'shift': [0., 0.], 'used_pairs': []}
        elif index in reachable:
            dx, dy = float(positions[index][0]), float(positions[index][1])
            matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)
            quality = alignment_quality(reference, images[index], matrix)
            details = {'method': 'chain:' + settings['method'], 'connected': True, 'shift': [dx, dy], 'used_pairs': used_pair_indices[index]}
        else:
            matrix = np.eye(3)
            quality = None
            details = {'method': 'disconnected', 'connected': False, 'shift': None, 'used_pairs': []}
        result.append({'frame_id': frame['id'], 'name': frame['name'], 'matrix': normalized_matrix(matrix, width, height), 'pixel_matrix': matrix.tolist(), 'quality': quality, 'details': details, 'seconds': time.monotonic() - started})
        if progress:
            progress(.85 + .1 * (index + 1) / count, '汇总位移 ' + frame['name'])
    disconnected = [frames[index]['name'] for index in range(count) if index not in reachable]
    return {'frames': result, 'reference': frames[reference_index]['id'], 'size': [width, height], 'settings': settings, 'chaining': True, 'applicable': not disconnected, 'disconnected_frames': disconnected, 'pairs': pairs}


def align_set(prepared, settings, progress=None, cancelled=None):
    if settings.get('chaining'):
        return chain_align_set(prepared, settings, progress, cancelled)
    images = prepared['images']
    reference_index = prepared['reference_index']
    reference = images[reference_index]
    width, height = prepared['size']
    result = []
    for index, (image, frame) in enumerate(zip(images, prepared['frames'])):
        if cancelled and cancelled():
            raise InterruptedError('配准任务已取消')
        started = time.monotonic()
        if index == reference_index:
            matrix, details = np.eye(3), {'method': 'reference'}
        elif settings['method'] == 'phase':
            matrix, details = phase_align(reference, image)
        elif settings['method'] == 'orb':
            matrix, details = orb_align(reference, image, settings)
        elif settings['method'] == 'ecc':
            matrix, details = ecc_align(reference, image, settings)
        else:
            matrix, details = mtb_align(reference, image)
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-8:
            raise v.Invalid('配准得到退化变换')
        quality = alignment_quality(reference, image, matrix)
        if quality['overlap'] < settings['minimum_overlap']:
            raise v.Invalid(frame['name'] + '配准覆盖不足，请检查参考帧和照片顺序')
        result.append({'frame_id': frame['id'], 'name': frame['name'], 'matrix': normalized_matrix(matrix, width, height), 'pixel_matrix': matrix.tolist(), 'quality': quality, 'details': details, 'seconds': time.monotonic() - started})
        if progress:
            progress(.35 + .55 * (index + 1) / len(images), '配准 ' + frame['name'])
    return {'frames': result, 'reference': prepared['frames'][reference_index]['id'], 'size': [width, height], 'settings': settings}
