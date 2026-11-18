import hashlib
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw
from . import validation as v


RING = 1
MAX_STAIN_SIDE = 64
MAX_IMAGE_EDGE = 512
VALID_THRESHOLD = .999
ALGORITHM = 'stain-patch-v1'
EDGE_ORDER = ('top', 'right', 'bottom', 'left')
EDGE_LABELS = {'top': '上边', 'right': '右边', 'bottom': '下边', 'left': '左边'}
RULE = '候选补片与污点框同尺寸、不旋转、不调色；补片外一圈与污点外一圈同位置比较。先取四边中差值最大的一边，最差边相同再取四边平均差最小的一边。'


def _integer_box(value, label, width, height, ring, maximum_side=None):
    if isinstance(value, dict):
        value = value.get('bounds', value.get('box'))
    if isinstance(value, list) and value and isinstance(value[0], (list, dict)):
        raise v.Invalid('本次最多修补一块污点，请只提供一个污点框')
    items = v.list_value(value, label, 4)
    if len(items) != 4:
        raise v.Invalid(label + '需要[x0, y0, x1, y1]四个像素坐标')
    x0, y0, x1, y1 = [v.integer(item, label + '坐标', 0, max(width, height)) for item in items]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise v.Invalid(label + '必须位于成图内，且右下边界大于左上边界')
    if maximum_side is not None and max(x1 - x0, y1 - y0) > maximum_side:
        raise v.Invalid(label + f'单边不能超过{maximum_side}像素')
    if ring and (x0 < RING or y0 < RING or x1 > width - RING or y1 > height - RING):
        raise v.Invalid(label + '需与成图边缘至少留1像素，才能比较框外一圈')
    return [x0, y0, x1, y1]


def parse_stain(value, width, height):
    return _integer_box(value, '污点框', width, height, True, MAX_STAIN_SIDE)


def parse_search(value, width, height):
    return _integer_box(value, '候选范围', width, height, False)


def full_search(width, height):
    return [0, 0, int(width), int(height)]


def _ring_bad_pixels(bad, x0, y0, x1, y1):
    """Count invalid pixels in the one-pixel ring just outside a box."""
    return int(bad[y0 - 1, x0 - 1:x1 + 1].sum()
               + bad[y1, x0 - 1:x1 + 1].sum()
               + bad[y0:y1, x0 - 1].sum()
               + bad[y0:y1, x1].sum())


def _bad_area(integral, x0, y0, x1, y1):
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def _line_scores(line, template, starts, offset=-RING):
    """Compare one ring segment at each candidate with the stain ring segment.

    The segments must match what _edge_values finally reports: horizontal
    sides include both corners, vertical sides do not (each ring pixel counts
    exactly once across the four sides).
    """
    length = template.shape[0]
    windows = line[np.asarray(starts)[:, None] + offset + np.arange(length)[None, :]]
    difference = np.abs(windows - template[None]).mean(axis=2)
    return difference.mean(axis=1)


def _edge_values(image, stain, patch):
    """Average absolute RGB difference for each side of the two outer rings.

    Corner pixels belong to the top and bottom sides, so every ring pixel is
    counted once. Differences are mean |ΔRGB| on the 0-1 display scale.
    """
    sx0, sy0, sx1, sy1 = stain
    px0, py0, px1, py1 = patch
    pairs = {
        'top': (image[sy0 - RING, sx0 - RING:sx1 + RING], image[py0 - RING, px0 - RING:px1 + RING]),
        'bottom': (image[sy1, sx0 - RING:sx1 + RING], image[py1, px0 - RING:px1 + RING]),
        'left': (image[sy0:sy1, sx0 - RING], image[py0:py1, px0 - RING]),
        'right': (image[sy0:sy1, sx1], image[py0:py1, px1]),
    }
    result = {}
    for name, (reference, candidate) in pairs.items():
        difference = np.abs(candidate - reference)
        channels = difference.mean(axis=0)
        result[name] = {
            'mean': float(difference.mean()),
            'channels': [float(channels[index]) for index in range(3)],
            'pixels': int(reference.shape[0]),
        }
    return result


def _reason(positions_in_search, ring_positions, outside_search, touches_stain, invalid_coverage):
    if positions_in_search <= 0:
        return '没有可用补片：候选范围比污点框还小，放不下同尺寸的补片。'
    if ring_positions <= 0:
        return '没有可用补片：补片和它外面一像素的一圈都要整个落在候选范围内，按这条候选范围没有足够位置。'
    parts = []
    if outside_search:
        parts.append(f'外圈超出候选范围的{outside_search}个位置已排除')
    if touches_stain:
        parts.append(f'补片或其外圈离污点框不足1像素的{touches_stain}个位置已排除')
    if invalid_coverage:
        parts.append(f'取到无效覆盖像素的{invalid_coverage}个位置已排除')
    detail = '；'.join(parts) if parts else '没有位置同时满足全部条件'
    return '没有可用补片：' + detail + '。请扩大候选范围或缩小污点框后重试。'


def analyze(image, valid, stain_value, search_value, result_id=None):
    """Find the clean same-size patch with the best four-edge outer-ring match.

    All bounds are integer pixel bounds [x0, y0, x1, y1] of the visible result
    image, with the right and bottom bounds exclusive.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise v.Invalid('成图必须是H×W×3图像')
    height, width = image.shape[:2]
    if max(width, height) > MAX_IMAGE_EDGE:
        raise v.Invalid(f'成图最长边为{max(width, height)}像素，超过本次修补上限{MAX_IMAGE_EDGE}像素；请把工作长边设为{MAX_IMAGE_EDGE}并重新合成')
    stain = parse_stain(stain_value, width, height)
    search = parse_search(full_search(width, height) if search_value is None else search_value, width, height)
    if valid is None:
        bad = np.zeros((height, width), np.uint8)
    else:
        valid = np.asarray(valid, dtype=np.float32)
        if valid.shape[:2] != (height, width):
            raise v.Invalid('有效覆盖图与成图尺寸不一致')
        bad = (valid < VALID_THRESHOLD).astype(np.int32)

    sx0, sy0, sx1, sy1 = stain
    bx0, by0, bx1, by1 = search
    patch_w, patch_h = sx1 - sx0, sy1 - sy0
    report = {
        'available': False,
        'algorithm': ALGORITHM,
        'stain': {'bounds': stain, 'width': patch_w, 'height': patch_h},
        'search': {'bounds': search, 'width': bx1 - bx0, 'height': by1 - by0},
        'edge_order': list(EDGE_ORDER),
        'edge_labels': EDGE_LABELS,
        'valid_threshold': VALID_THRESHOLD,
        'limits': {'stains': 1, 'stain_side': MAX_STAIN_SIDE, 'image_edge': MAX_IMAGE_EDGE},
        'rule': RULE,
        'stain_clearance': '补片及其外圈与污点框之间至少留1像素空隙，既不共用像素也不贴边。',
        'ring_scope': '每条边取补片外一圈与污点外一圈的同位置像素：上下边含左右两个角，左右边不含角，整圈每个像素只计一次。',
        'comparison_space': '显示sRGB像素|x-y|按RGB三通道平均，0–1尺度，另给出×255数值。',
    }
    stain_ring_bad = _ring_bad_pixels(bad, sx0, sy0, sx1, sy1)
    if stain_ring_bad:
        report.update(reason=f'污点框外一圈含{stain_ring_bad}个无效覆盖像素，无法与补片比对颜色。', candidate_count=0, positions_in_search=0)
        return report

    # A patch top-left at (x, y) needs its own 1px outer ring inside the search
    # box, and a 1px clear gap all round the stain box.
    x_start, x_stop = bx0 + RING, bx1 - patch_w - RING
    y_start, y_stop = by0 + RING, by1 - patch_h - RING
    xs = np.arange(x_start, x_stop + 1, dtype=np.int64) if x_stop >= x_start else np.empty(0, np.int64)
    ys = np.arange(y_start, y_stop + 1, dtype=np.int64) if y_stop >= y_start else np.empty(0, np.int64)
    ring_positions = int(xs.size * ys.size)
    positions_in_search = max(0, bx1 - bx0 - patch_w + 1) * max(0, by1 - by0 - patch_h + 1)
    outside_search = max(0, positions_in_search - ring_positions)
    if not ring_positions:
        report.update(reason=_reason(positions_in_search, 0, outside_search, 0, 0), candidate_count=0, positions_in_search=positions_in_search,
                      rejections={'outer_outside_search': outside_search, 'touches_stain': 0, 'invalid_coverage': 0})
        return report

    xg, yg = np.meshgrid(xs, ys)
    # The patch and its outer ring must not touch the stain box: keep at least
    # one untouched pixel between them (so they share no pixel and no edge).
    outer_left, outer_top = xg - RING, yg - RING
    outer_right, outer_bottom = xg + patch_w + RING, yg + patch_h + RING
    clear_of_stain = (outer_right <= sx0 - 1) | (outer_left >= sx1 + 1) | (outer_bottom <= sy0 - 1) | (outer_top >= sy1 + 1)
    integral = np.zeros((height + 1, width + 1), np.int32)
    integral[1:, 1:] = bad.cumsum(axis=0).cumsum(axis=1)
    invalid = _bad_area(integral, xg - RING, yg - RING, xg + patch_w + RING, yg + patch_h + RING) > 0
    touches_stain = int((~clear_of_stain).sum())
    invalid_coverage = int((clear_of_stain & invalid).sum())
    allowed = clear_of_stain & ~invalid
    evaluated = int(allowed.sum())
    report.update(candidate_count=evaluated, positions_in_search=positions_in_search,
                  rejections={'outer_outside_search': outside_search, 'touches_stain': touches_stain, 'invalid_coverage': invalid_coverage})
    if not evaluated:
        report['reason'] = _reason(positions_in_search, ring_positions, outside_search, touches_stain, invalid_coverage)
        return report

    top_template = image[sy0 - RING, sx0 - RING:sx1 + RING]
    bottom_template = image[sy1, sx0 - RING:sx1 + RING]
    left_template = image[sy0:sy1, sx0 - RING]
    right_template = image[sy0:sy1, sx1]
    edge_maps = {name: np.empty((ys.size, xs.size), np.float32) for name in EDGE_ORDER}
    for row_index, y in enumerate(ys):
        edge_maps['top'][row_index] = _line_scores(image[y - RING], top_template, xs)
        edge_maps['bottom'][row_index] = _line_scores(image[y + patch_h], bottom_template, xs)
    for column_index, x in enumerate(xs):
        edge_maps['left'][:, column_index] = _line_scores(image[:, x - RING], left_template, ys, 0)
        edge_maps['right'][:, column_index] = _line_scores(image[:, x + patch_w], right_template, ys, 0)

    worst = np.maximum.reduce([edge_maps[name] for name in EDGE_ORDER])
    mean = sum(edge_maps[name] for name in EDGE_ORDER) / len(EDGE_ORDER)
    worst = np.where(allowed, worst, np.inf).ravel()
    mean = np.where(allowed, mean, np.inf).ravel()
    flat_x, flat_y = xg.ravel(), yg.ravel()
    # lexsort keys are applied last-to-first: worst edge, then mean, then y, then x.
    order = np.lexsort((flat_x, flat_y, mean, worst))
    chosen = int(order[0])
    best_x, best_y = int(flat_x[chosen]), int(flat_y[chosen])
    patch = [best_x, best_y, best_x + patch_w, best_y + patch_h]
    edges = _edge_values(image, stain, patch)
    edge_values = {name: edges[name]['mean'] for name in EDGE_ORDER}
    worst_edge = max(EDGE_ORDER, key=lambda name: edge_values[name])
    report.update({
        'available': True,
        'patch': {'bounds': patch, 'width': patch_w, 'height': patch_h, 'outer_bounds': [best_x - RING, best_y - RING, best_x + patch_w + RING, best_y + patch_h + RING]},
        'edge_differences': {name: round(edge_values[name], 8) for name in EDGE_ORDER},
        'edge_differences_255': {name: round(edge_values[name] * 255, 3) for name in EDGE_ORDER},
        'edge_channels': {name: {
            'mean': round(edges[name]['mean'], 8),
            'mean_255': round(edges[name]['mean'] * 255, 3),
            'channels': {channel: round(edges[name]['channels'][index], 8) for index, channel in enumerate('rgb')},
            'channels_255': {channel: round(edges[name]['channels'][index] * 255, 3) for index, channel in enumerate('rgb')},
            'pixels': edges[name]['pixels']} for name in EDGE_ORDER},
        'worst_edge': worst_edge,
        'worst_edge_label': EDGE_LABELS[worst_edge],
        'worst_difference': round(edge_values[worst_edge], 8),
        'worst_difference_255': round(edge_values[worst_edge] * 255, 3),
        'edges_good_except_worst': all(edge_values[name] <= edge_values[worst_edge] * .8 for name in EDGE_ORDER if name != worst_edge),
        'mean_difference': round(float(np.mean(list(edge_values.values()))), 8),
        'mean_difference_255': round(float(np.mean(list(edge_values.values()))) * 255, 3),
        'ring_pixels': int(2 * (patch_w + 2 * RING) + 2 * patch_h),
    })
    report['alternatives'] = _alternatives(order, flat_x, flat_y, worst, mean, edge_maps, patch_w, patch_h)
    report['scheme_id'] = scheme_digest(result_id, stain, search, patch, report['edge_differences'])
    return report


def _alternatives(order, flat_x, flat_y, worst, mean, edge_maps, patch_w, patch_h, limit=4):
    rows = []
    for index in np.asarray(order)[:limit + 1]:
        index = int(index)
        if not np.isfinite(worst[index]):
            continue
        x, y = int(flat_x[index]), int(flat_y[index])
        rows.append({
            'bounds': [x, y, x + patch_w, y + patch_h],
            'worst_difference': round(float(worst[index]), 8),
            'worst_difference_255': round(float(worst[index]) * 255, 3),
            'mean_difference': round(float(mean[index]), 8),
            'mean_difference_255': round(float(mean[index]) * 255, 3),
            'edge_differences': {name: round(float(edge_maps[name].ravel()[index]), 8) for name in EDGE_ORDER},
        })
    if len(rows) > 1:
        rows = [row for row in rows if row['bounds'] != rows[0]['bounds']][:limit]
    else:
        rows = []
    return rows


def scheme_digest(result_id, stain, search, patch, edge_values):
    payload = {'algorithm': ALGORITHM, 'result_id': result_id, 'stain': list(stain), 'search': list(search), 'patch': list(patch), 'edges': edge_values}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def fill_image(image, stain_bounds, patch_bounds):
    x0, y0, x1, y1 = stain_bounds
    px0, py0, px1, py1 = patch_bounds
    if (x1 - x0, y1 - y0) != (px1 - px0, py1 - py0):
        raise v.Invalid('补片尺寸必须与污点框相同')
    result = np.asarray(image, dtype=np.float32).copy()
    result[y0:y1, x0:x1] = result[py0:py1, px0:px1]
    return result


def changed_outside(image, filled, stain_bounds):
    difference = np.any(np.abs(np.asarray(filled) - np.asarray(image)) > 1e-6, axis=2)
    difference[stain_bounds[1]:stain_bounds[3], stain_bounds[0]:stain_bounds[2]] = False
    return int(difference.sum())


def suggest(image):
    """Heuristically pre-fill a stain box; the operator still confirms it."""
    image = np.asarray(image, dtype=np.float32)
    height, width = image.shape[:2]
    gray = np.clip(image[:, :, :3] @ np.array([.2126, .7152, .0722], np.float32), 0, 1)
    magnitude = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 9))
    mask = (magnitude > .12).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    rows = []
    for index in range(1, count):
        x, y, w, h, area = [int(item) for item in stats[index]]
        if area < 12:
            continue
        room = x >= 1 and y >= 1 and x + w + 2 <= width and y + h + 2 <= height
        usable = w <= MAX_STAIN_SIDE and h <= MAX_STAIN_SIDE and room
        score = float(magnitude[labels == index].sum())
        rows.append({'bounds': [x, y, x + w, y + h], 'area': area, 'usable': usable, 'score': round(score, 6)})
    ranked = sorted([row for row in rows if row['usable']], key=lambda row: row['score'], reverse=True)
    if not ranked:
        reason = '没有找到足够鲜明的异常块，请手工框选。' if not rows else f'检出异常块，但单边超过{MAX_STAIN_SIDE}像素或太靠成图边缘，请手工框选。'
        return {'detected': False, 'reason': reason, 'candidates': rows[:8], 'stain': None, 'search': None}
    best = ranked[0]
    side = max(best['bounds'][2] - best['bounds'][0], best['bounds'][3] - best['bounds'][1])
    margin = max(32, side * 2)
    x0, y0, x1, y1 = best['bounds']
    search = [max(0, x0 - margin), max(0, y0 - margin), min(width, x1 + margin), min(height, y1 + margin)]
    if (search[2] - search[0]) * (search[3] - search[1]) < (side + 2) ** 2 * 2:
        search = [0, 0, width, height]
    return {'detected': True, 'stain': best['bounds'], 'search': search, 'candidates': rows[:8], 'note': '按局部深色异常块预填，需人工确认框位。'}


def overlay_image(image, stain_bounds, search_bounds, patch_bounds=None):
    raster = Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)).convert('RGBA')
    layer = Image.new('RGBA', raster.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    width, height = raster.size

    def rectangle(bounds, color, fill=None):
        x0, y0, x1, y1 = bounds
        draw.rectangle((max(0, x0), max(0, y0), min(width - 1, x1 - 1), min(height - 1, y1 - 1)), outline=color, fill=fill, width=2)

    def label(bounds, text, color, above=True):
        x = min(max(0, bounds[0]), max(0, width - len(text) * 7 - 6))
        y = bounds[1] - 14 if above and bounds[1] >= 14 else bounds[1] + 2
        y = min(max(0, y), max(0, height - 13))
        draw.rectangle((x, y, x + len(text) * 7 + 5, y + 12), fill=color)
        draw.text((x + 3, y + 1), text, fill=(248, 250, 240, 255))

    rectangle(search_bounds, (214, 178, 105, 255))
    label(search_bounds, 'SEARCH', (176, 140, 70, 235))
    rectangle(stain_bounds, (226, 86, 54, 255), (226, 86, 54, 66))
    label(stain_bounds, 'STAIN', (190, 66, 40, 235))
    if patch_bounds is not None:
        rectangle(patch_bounds, (49, 139, 117, 255))
        label(patch_bounds, 'PATCH', (30, 92, 78, 235), above=False)
    return np.asarray(Image.alpha_composite(raster, layer).convert('RGB'), np.float32) / 255
