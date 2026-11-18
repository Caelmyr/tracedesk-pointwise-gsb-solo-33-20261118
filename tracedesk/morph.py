"""两条等节点闭环之间的过渡轮廓（徽记第三版）。

配法只允许沿环顺序的循环平移与整体反向，不允许拆散环上相邻的角；
对应角在同一进度上线性插值。一张配法只有在 25%、50%、75% 三张中间稿上
都保持简单闭环（无重合节点、无边相交、无非相邻边相碰）时才可用。
合格配法按（最远角移动量，所有角总移动量）字典序择优，枚举顺序作为平局依据。
"""
import math

from image_lab.validation import Invalid

STEPS = (0.25, 0.5, 0.75)
MIN_NODES = 3
MAX_NODES = 12


def select_pair(drawing, old_id, new_id):
    paths = drawing.get('paths') if isinstance(drawing, dict) else None
    if not isinstance(paths, list):
        raise Invalid('图稿数据无效')
    by_id = {q.get('id'): q for q in paths if isinstance(q, dict)}
    old = by_id.get(old_id)
    new = by_id.get(new_id)
    if old is None or new is None:
        raise Invalid('请选择旧版与新版两条路径')
    if old_id == new_id:
        raise Invalid('请选择两条不同的闭环路径')
    for label, q in (('旧版', old), ('新版', new)):
        if not q.get('closed'):
            raise Invalid(f'{label}路径必须是闭合路径')
        n = len(q.get('points', []))
        if not MIN_NODES <= n <= MAX_NODES:
            raise Invalid(f'{label}路径需为{MIN_NODES}到{MAX_NODES}个节点的闭环（当前{n}点）')
    n_old, n_new = len(old['points']), len(new['points'])
    if n_old != n_new:
        raise Invalid(f'两条路径节点数不同（旧版{n_old}点、新版{n_new}点），无法逐角配对')
    return old, new


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _same_point(a, b, tol):
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def _on_segment(a, b, p, eps):
    if abs(_cross(a, b, p)) > eps:
        return False
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)


def _strict_on(a, b, p, eps, tol):
    """p 落在 ab 线段内部（不含端点）。"""
    return _on_segment(a, b, p, eps) and not _same_point(p, a, tol) and not _same_point(p, b, tol)


def _contact(a, b, c, d, eps, tol):
    """两线段接触情况。

    返回 None（不接触）、('point', 交点)（仅单点接触）或 ('overlap', None)
    （共线且有正长度重叠，或端点落在另一线段内部）。
    """
    d1 = _cross(c, d, a)
    d2 = _cross(c, d, b)
    d3 = _cross(a, b, c)
    d4 = _cross(a, b, d)
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) \
            and ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        t = d3 / (d3 - d4)
        return 'point', (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    if abs(d1) <= eps and abs(d2) <= eps and abs(d3) <= eps and abs(d4) <= eps:
        if not (_on_segment(c, d, a, eps) or _on_segment(c, d, b, eps)
                or _on_segment(a, b, c, eps) or _on_segment(a, b, d, eps)):
            return None
        if _strict_on(c, d, a, eps, tol) or _strict_on(c, d, b, eps, tol) \
                or _strict_on(a, b, c, eps, tol) or _strict_on(a, b, d, eps, tol):
            return 'overlap', None
    hits = []
    for p in (a, b, c, d):
        if (p is a or p is b) and _on_segment(c, d, p, eps):
            hits.append(p)
        elif (p is c or p is d) and _on_segment(a, b, p, eps):
            hits.append(p)
    if not hits:
        return None
    # 端点落在另一线段内部即构成正长度相碰（T 型接触），按重叠处理
    if _strict_on(c, d, a, eps, tol) or _strict_on(c, d, b, eps, tol) \
            or _strict_on(a, b, c, eps, tol) or _strict_on(a, b, d, eps, tol):
        return 'overlap', None
    unique = []
    for p in hits:
        if not any(_same_point(p, q, tol) for q in unique):
            unique.append(p)
    if len(unique) >= 2:
        return 'overlap', None
    return 'point', unique[0]


def frame_valid(points, eps, tol):
    """中间稿必须是无自交、无重节点的简单闭环。"""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if _same_point(points[i], points[j], tol):
                return False
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            c, d = points[j], points[(j + 1) % n]
            hit = _contact(a, b, c, d, eps, tol)
            if hit is None:
                continue
            adjacent = (j == i + 1) or (i == 0 and j == n - 1)
            if not adjacent:
                return False
            shared = b if j == i + 1 else points[0]
            if hit[0] != 'point' or not _same_point(hit[1], shared, tol):
                return False
    return True


def _ordered_target(b_points, shift, reversed_order, n):
    if reversed_order:
        return [b_points[(shift - i) % n] for i in range(n)]
    return [b_points[(shift + i) % n] for i in range(n)]


def _interpolate(a_points, b_points, t):
    return [[(1 - t) * x + t * u, (1 - t) * y + t * v]
            for (x, y), (u, v) in zip(a_points, b_points)]


def _round_points(points):
    return [[round(x, 4), round(y, 4)] for x, y in points]


def _round_point(p):
    return [round(p[0], 4), round(p[1], 4)]


def morph_paths(drawing, old_id, new_id):
    old, new = select_pair(drawing, old_id, new_id)
    a_points, b_points = old['points'], new['points']
    n = len(a_points)
    scale = max(float(drawing.get('width', 1)), float(drawing.get('height', 1)), 1.0)
    eps = 1e-9 * scale * scale
    tol = 1e-6 * scale
    candidates = []
    for shift in range(n):
        for reversed_order in (False, True):
            target = _ordered_target(b_points, shift, reversed_order, n)
            distances = [math.hypot(u - x, v - y)
                         for (x, y), (u, v) in zip(a_points, target)]
            frames = [_interpolate(a_points, target, t) for t in STEPS]
            if all(frame_valid(frame, eps, tol) for frame in frames):
                candidates.append((max(distances), sum(distances), shift,
                                   reversed_order, target, distances, frames))
    if not candidates:
        raise Invalid(
            f'找不到可用配法：{2 * n}种起点/方向配法在25%、50%、75%中间稿上'
            '均出现边相交、非相邻边相碰或节点重合，未向图稿添加内容')
    max_move, total_move, shift, reversed_order, target, distances, frames = min(
        candidates, key=lambda r: (r[0], r[1], r[2], r[3]))
    return {
        'old_id': old_id,
        'new_id': new_id,
        'node_count': n,
        'match': {'shift': shift, 'reversed': reversed_order},
        'max_move': round(max_move, 3),
        'total_move': round(total_move, 3),
        'pairs': [{'from': _round_point(a),
                   'to': _round_point(b),
                   'distance': round(d, 3)}
                  for a, b, d in zip(a_points, target, distances)],
        'frames': [{'t': t, 'points': _round_points(frame)}
                   for t, frame in zip(STEPS, frames)],
        'candidate_count': 2 * n,
        'valid_candidate_count': len(candidates),
    }


def add_morph_path(drawing, body):
    old, new = select_pair(drawing, body.get('old_id'), body.get('new_id'))
    try:
        t = float(body.get('t'))
    except (TypeError, ValueError):
        raise Invalid('过渡进度无效')
    if not math.isfinite(t) or not any(abs(t - s) <= 1e-9 for s in STEPS):
        raise Invalid('仅可添加25%、50%、75%三张中间稿之一')
    result = morph_paths(drawing, body.get('old_id'), body.get('new_id'))
    frame = next(f for f in result['frames'] if abs(f['t'] - t) <= 1e-9)
    from .core import edit
    default_name = f'{old["name"]} → {new["name"]} 过渡{int(round(t * 100))}%'
    name = str(body.get('name') or default_name)[:80]
    edit(drawing, {'action': 'add', 'name': name,
                   'points': frame['points'], 'closed': True})
    return result
