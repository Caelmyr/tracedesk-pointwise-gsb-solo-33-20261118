import html
import math

from image_lab.validation import Invalid

from .core import integer

MAX_PATHS = 6
MAX_CLOSED_NODES = 10
ORIGIN = (0.0, 0.0)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def check_scope(paths):
    n = len(paths)
    if n == 0:
        raise Invalid('图稿还没有路径，无法安排走笔')
    if n > MAX_PATHS:
        raise Invalid(f'一次最多处理{MAX_PATHS}条路径，当前{n}条')
    for i, q in enumerate(paths):
        if q.get('closed') and len(q['points']) > MAX_CLOSED_NODES:
            raise Invalid(
                f'闭环“{q.get("name", i + 1)}”有{len(q["points"])}个节点，'
                f'走笔安排每条闭环最多{MAX_CLOSED_NODES}个节点'
            )


def path_options(paths, i):
    """可选的画笔画法：(起笔点, 收笔点, 反向标记, 起笔节点序号, 是否挪动闭环起点)。"""
    q = paths[i]
    pts = q['points']
    if q.get('closed'):
        return [(pts[k], pts[k - 1], False, k, k != 0) for k in range(len(pts))]
    return [
        (pts[0], pts[-1], False, 0, False),
        (pts[-1], pts[0], True, len(pts) - 1, False),
    ]


def baseline_travel(paths):
    """按SVG现有的路径顺序、原方向、原起点的空走距离（含到第一笔）。"""
    total = dist(ORIGIN, paths[0]['points'][0])
    for a, b in zip(paths, paths[1:]):
        end = a['points'][-1]
        if a.get('closed'):
            end = a['points'][0]
        total += dist(end, b['points'][0])
    return total


def build_plan(p, revision):
    """穷举顺序/方向/闭环起点（至多挪动一个闭环），求空走距离最短的走笔方案。"""
    rev = integer(revision, 0, 10**9, '修订号')
    paths = p['paths']
    check_scope(paths)
    n = len(paths)
    options = [path_options(paths, i) for i in range(n)]
    original_order = tuple(range(n))
    best = None

    def consider(cost, order, picks, moves, reversals):
        nonlocal best
        key = (
            round(cost, 4),
            moves,
            reversals,
            sum((pos - idx) ** 2 for pos, idx in zip(order, original_order)),
            order,
            picks,
        )
        if best is None or key < best[0]:
            best = (key, cost, order, picks)

    def search(used, order, picks, moved, reversals, current_end, cost):
        if used == (1 << n) - 1:
            consider(cost, tuple(order), tuple(picks), moved, reversals)
            return
        for i in range(n):
            if used & (1 << i):
                continue
            for k, (start, end, reversed_flag, node_idx, moved_flag) in enumerate(options[i]):
                if moved and moved_flag:
                    continue
                step = dist(current_end, start)
                search(
                    used | (1 << i),
                    order + [i],
                    picks + [k],
                    moved or moved_flag,
                    reversals + (1 if reversed_flag else 0),
                    end,
                    cost + step,
                )

    search(0, [], [], False, 0, ORIGIN, 0.0)
    _, total, order, picks = best

    steps = []
    pen_up = []
    cursor = ORIGIN
    for position, (i, k) in enumerate(zip(order, picks)):
        q = paths[i]
        start, end, reversed_flag, node_idx, moved_flag = options[i][k]
        gap = dist(cursor, start)
        pen_up.append({
            'from': [round(cursor[0], 4), round(cursor[1], 4)],
            'to': [round(start[0], 4), round(start[1], 4)],
            'distance': round(gap, 2),
        })
        steps.append({
            'order': position + 1,
            'path_id': q['id'],
            'name': q['name'],
            'closed': bool(q.get('closed')),
            'reversed': bool(reversed_flag),
            'start_index': node_idx,
            'start_moved': bool(moved_flag),
            'start': [round(start[0], 4), round(start[1], 4)],
            'end': [round(end[0], 4), round(end[1], 4)],
            'air_distance': round(gap, 2),
        })
        cursor = end

    old_total = baseline_travel(paths)
    return {
        'revision': rev,
        'path_count': n,
        'origin': [0.0, 0.0],
        'original_travel': round(old_total, 2),
        'planned_travel': round(total, 2),
        'saved': round(old_total - total, 2),
        'moved_closed_count': sum(1 for s in steps if s['start_moved']),
        'steps': steps,
        'pen_up': pen_up,
    }


def ordered_points(q, step):
    pts = q['points']
    if q.get('closed'):
        k = step['start_index']
        return pts[k:] + pts[:k]
    if step['reversed']:
        return list(reversed(pts))
    return list(pts)


def plan_svg(p, plan):
    """按预览的顺序、方向、起点写出SVG；路径节点与闭合关系保持不变。"""
    by_id = {q['id']: q for q in p['paths']}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{p["width"]}" height="{p["height"]}" '
        f'viewBox="0 0 {p["width"]} {p["height"]}">',
        f'<title>{html.escape(p["name"])} · 走笔安排</title>',
        f'<g fill="none" stroke="{p["stroke"]}" stroke-width="{p["line_width"]}" '
        'stroke-linecap="round" stroke-linejoin="round">',
    ]
    for step in plan['steps']:
        q = by_id[step['path_id']]
        pts = ordered_points(q, step)
        d = 'M ' + ' L '.join(f'{float(x):.4f} {float(y):.4f}' for x, y in pts)
        if q.get('closed'):
            d += ' Z'
        lines.append(
            f'<path d="{d}" data-path-id="{q["id"]}" data-plan-order="{step["order"]}"/>'
        )
    lines += ['</g>', '</svg>']
    return '\n'.join(lines).encode()
