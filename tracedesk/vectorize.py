"""干净单色线稿的自动单线矢量化。

约束：底图最长边不超过256像素、笔画约3至15像素宽的不透明线稿。
流程：阈值二值化 -> Zhang-Suen 细化取笔画中线 -> 骨架拆图（分叉共享节点、
真闭环、连通域之间不相连）-> 忠实/简洁两种取向输出与现有描线同构的路径。
"""
import math
import numpy as np
from scipy import ndimage
from image_lab.validation import Invalid
from .core import integer, png

MAX_EDGE = 256
FAITHFUL_EPS = 0.5
SIMPLE_EPS = 1.4
TIP_LIMIT = 3.0
MAX_PATHS = 500
MAX_POINTS = 4096


def binarize(a, threshold):
    """RGBA 线稿合成到白底后按灰度阈值取笔画掩膜（暗于阈值视为笔画）。"""
    a = np.asarray(a)
    if a.ndim != 3 or a.shape[2] != 4:
        raise Invalid('底图格式无效')
    rgb = a[..., :3].astype(np.float64)
    alpha = a[..., 3:4].astype(np.float64) / 255.0
    comp = rgb * alpha + 255.0 * (1.0 - alpha)
    gray = comp[..., 0] * 0.299 + comp[..., 1] * 0.587 + comp[..., 2] * 0.114
    return gray < float(threshold)


def mask_png(a, threshold):
    m = binarize(a, integer(threshold, 0, 255, '黑白阈值'))
    h, w = m.shape
    out = np.full((h, w, 4), 255, dtype=np.uint8)
    out[m] = (0, 0, 0, 255)
    return png(out)


def zhang_suen(mask):
    """Zhang-Suen 细化，返回与输入等大的单像素骨架布尔图。"""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p2 = np.zeros_like(img); p2[:-2, 1:-1] = img[1:-1, 1:-1]
            p3 = np.zeros_like(img); p3[:-2, 2:] = img[1:-1, 1:-1]
            p4 = np.zeros_like(img); p4[1:-1, 2:] = img[1:-1, 1:-1]
            p5 = np.zeros_like(img); p5[2:, 2:] = img[1:-1, 1:-1]
            p6 = np.zeros_like(img); p6[2:, 1:-1] = img[1:-1, 1:-1]
            p7 = np.zeros_like(img); p7[2:, :-2] = img[1:-1, 1:-1]
            p8 = np.zeros_like(img); p8[1:-1, :-2] = img[1:-1, 1:-1]
            p9 = np.zeros_like(img); p9[:-2, :-2] = img[1:-1, 1:-1]
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            transitions = sum((cur == 0) & (nxt == 1)
                              for cur, nxt in zip(neighbors, neighbors[1:] + [p2]))
            degree = sum(neighbors)
            cond = (img == 1) & (degree >= 2) & (degree <= 6) & (transitions == 1)
            if step == 0:
                cond &= (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond &= (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            if cond.any():
                img[cond] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1].astype(bool)


def _xy(p):
    return [p[1], p[0]]


def _isolated_arms(ring):
    """环中两侧邻居都为背景的前景像素数：真分叉>=3；环自接触点为0。"""
    n = len(ring)
    return sum(v and not ring[(i - 1) % n] and not ring[(i + 1) % n]
               for i, v in enumerate(ring))


def _length(points):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def walk_chain(adj, start, allowed, stop_set):
    """从 start 沿骨架行走到 stop_set 中的端点；带 visited 的顺序行走。"""
    seq = [start]
    visited = {start}
    prev = None
    cur = start
    guard = 0
    limit = len(allowed) + 2
    while guard <= limit:
        guard += 1
        nxts = [q for q in adj[cur] if q in allowed and q != prev and q not in visited]
        if not nxts:
            break
        nxt = nxts[0]
        seq.append(nxt)
        visited.add(nxt)
        if nxt in stop_set:
            break
        prev, cur = cur, nxt
    return seq


def trace_ring(adj, start, allowed):
    """从环上一点出发，深度优先找回到 start 的完整环（环含对角台阶）。

    返回环上像素序列；若无法闭合返回 None。
    """
    stack = [(start, [start], {start})]
    limit = len(allowed)
    while stack:
        cur, path, visited = stack.pop()
        if len(path) > limit:
            continue
        for q in adj[cur]:
            if q not in allowed:
                continue
            if q == start and len(path) >= 3:
                return path
            if q in visited:
                continue
            stack.append((q, path + [q], visited | {q}))
    return None


def graph_edges(skel, mask):
    """把骨架拆成边：分叉处多条边共享同一节点坐标，闭环置 closed。"""
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    padded = np.pad(skel, 1)
    ys, xs = np.nonzero(skel)
    pixels = list(zip(ys.tolist(), xs.tolist()))
    adj = {}
    deg8 = {}
    arms = {}
    for y, x in pixels:
        ns = [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
              if (dy or dx) and padded[y + 1 + dy, x + 1 + dx]]
        adj[(y, x)] = ns
        deg8[(y, x)] = len(ns)
        ring = [int(padded[y + 1 + dy, x + 1 + dx])
                for dy, dx in ((-1, 0), (-1, 1), (0, 1), (1, 1),
                               (1, 0), (1, -1), (0, -1), (-1, -1))]
        arms[(y, x)] = _isolated_arms(ring)
    endpoints = {p for p in pixels if deg8[p] == 1}
    # 四邻域用于臂分量统计（八邻域下分叉周围的臂首像素对角相邻会被误并）
    adj4 = {p: [(p[0] + dy, p[1] + dx) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if padded[p[0] + 1 + dy, p[1] + 1 + dx]] for p in pixels}
    # 局部候选（含环自接触点）先按八连通聚簇，再用全局拓扑判定真分叉
    raw = {p for p in pixels if deg8[p] >= 3 and arms[p] >= 3}
    junc = set()
    raw_seen = set()
    for seed in raw:
        if seed in raw_seen:
            continue
        stack = [seed]
        cluster = []
        raw_seen.add(seed)
        while stack:
            p = stack.pop()
            cluster.append(p)
            for q in adj[p]:
                if q in raw and q not in raw_seen:
                    raw_seen.add(q)
                    stack.append(q)
        cset = set(cluster)
        boundary = set()
        for p in cluster:
            for q in adj[p]:
                if q not in cset:
                    boundary.add(q)
        # 删除整个簇后，簇外骨架被分成多少个连通分量，即该节点的独立臂数。
        # 真分叉/交叉>=3条互不相通的臂；环捏合点的两条臂沿环相通，只有<=2个分量。
        arm_count = 0
        seen_arm = set()
        for bnd in boundary:
            if bnd in seen_arm:
                continue
            arm_count += 1
            stk = [bnd]
            seen_arm.add(bnd)
            while stk:
                q = stk.pop()
                for n in adj4[q]:
                    if n not in cset and n not in seen_arm:
                        seen_arm.add(n)
                        stk.append(n)
        if arm_count >= 3:
            junc.update(cluster)

    # 分叉像素按八连通聚成簇，簇质心作为所有相接边的共享节点
    cluster_of = {}
    clusters = []
    seen = set()
    for seed in junc:
        if seed in seen:
            continue
        cid = len(clusters)
        stack = [seed]
        members = []
        seen.add(seed)
        while stack:
            p = stack.pop()
            members.append(p)
            cluster_of[p] = cid
            for q in adj[p]:
                if q in junc and q not in seen:
                    seen.add(q)
                    stack.append(q)
        cy = sum(p[0] for p in members) / len(members)
        cx = sum(p[1] for p in members) / len(members)
        clusters.append({'id': cid, 'points': members,
                         'xy': [round(cx, 1), round(cy, 1)],
                         'comp': int(labels[seed])})

    edges = []
    used = set()
    direct_pairs = set()

    def add(points, closed, comp, nodes):
        if len(points) >= 2:
            edges.append({'pts': points, 'closed': closed, 'comp': comp, 'nodes': nodes})

    def trace_arm(start, blocked):
        """从簇边界 start 顺序行走到出口（确定性，无回溯，避免环上指数爆炸）。

        对角台阶会带来额外的回头邻居，用 visited 集合排除；每个环点保留两个
        前进方向，行走时取未访问的邻居。返回 (类型, 终端像素, 像素路径)。
        类型：'j' 到另一分叉簇；'loop' 回本簇边界；'e' 到末梢或走完。
        """
        own_boundary = set()
        for pt in blocked:
            for q in adj[pt]:
                if q not in blocked:
                    own_boundary.add(q)
        path = [start]
        visited = {start}
        # 初始方向：背离所属簇质心，避免第一步按邻接枚举顺序拐入侧臂
        cy = sum(pt[0] for pt in blocked) / len(blocked)
        cx = sum(pt[1] for pt in blocked) / len(blocked)
        prev = (round(2 * cy - start[0]), round(2 * cx - start[1]))
        cur = start
        guard = 0
        while guard <= len(pixels):
            guard += 1
            nxts = [q for q in adj[cur] if q != prev and q not in blocked and q not in visited]
            if not nxts:
                return 'e', cur, path
            junc_nxt = next((q for q in nxts if q in junc), None)
            if junc_nxt is not None:
                return 'j', junc_nxt, path
            loop_nxt = next((q for q in nxts if q in own_boundary), None)
            if loop_nxt is not None and len(path) >= 3:
                path.append(loop_nxt)
                return 'loop', loop_nxt, path
            # 选与当前走向最一致的邻居，避免在对角台阶处拐入侧臂
            dy0, dx0 = cur[0] - prev[0], cur[1] - prev[1]
            nxt = max(nxts, key=lambda q: (q[0] - cur[0]) * dy0 + (q[1] - cur[1]) * dx0)
            path.append(nxt)
            visited.add(nxt)
            if nxt in endpoints:
                return 'e', nxt, path
            prev, cur = cur, nxt
        return 'e', cur, path

    # 每个分叉簇：从每个边界像素沿一条臂追踪，环的两条臂通过同路径终点配对
    for c in clusters:
        cset = set(c['points'])
        boundary = []
        for pt in c['points']:
            for q in adj[pt]:
                if q not in cset and q not in boundary:
                    boundary.append(q)
        traced = {}
        for b0 in boundary:
            traced[b0] = trace_arm(b0, cset)
            used.update(traced[b0][2])
        emitted = set()
        seen_end = set()
        # 环的两条半圈各自 trace 回本簇边界，合并成一条完整闭环
        loop_paths = [path for kind, end, path in traced.values() if kind == 'loop']
        if loop_paths:
            merged = []
            for path in loop_paths:
                merged.extend(path)
            ring_seq = [list(c['xy'])] + [_xy(q) for q in merged]
            if _length(ring_seq) >= 4.0:
                add(ring_seq, True, c['comp'], [('j', c['id']), ('j', c['id'])])
            for b0, (kind, end, path) in traced.items():
                if kind == 'loop':
                    emitted.add(b0)
        for b0 in boundary:
            if b0 in emitted:
                continue
            kind, end, path = traced[b0]
            dedup_key = (kind, end)
            if kind == 'e' and dedup_key in seen_end:
                emitted.add(b0)
                continue
            seen_end.add(dedup_key)
            seq = [list(c['xy'])] + [_xy(q) for q in path]
            if kind == 'loop':
                emitted.add(b0)
                continue
            if kind == 'j':
                d = cluster_of[end]
                if d == c['id']:
                    emitted.add(b0)
                    continue
                key = tuple(sorted((c['id'], d)))
                if key not in direct_pairs:
                    direct_pairs.add(key)
                    add([list(c['xy']), list(clusters[d]['xy'])], False, c['comp'],
                        [('j', c['id']), ('j', d)])
                emitted.add(b0)
                continue
            add(seq, False, c['comp'], [('j', c['id']), ('e', end)])
            emitted.add(b0)

    # 不含分叉的连通域：开口曲线或纯闭环
    handled = {p for p in used} | set(cluster_of)
    for comp_id in sorted({int(labels[p]) for p in pixels if p not in junc}):
        comp_pixels = [p for p in pixels
                       if int(labels[p]) == comp_id and p not in junc and p not in handled]
        if not comp_pixels:
            continue
        comp_set = set(comp_pixels)
        starts = [p for p in comp_pixels if p in endpoints]
        if not starts:
            # 无端点即纯闭环（环上对角台阶的度数3像素已在分叉判定中被排除）
            s = comp_pixels[0]
            ring = trace_ring(adj, s, comp_set)
            if ring is not None:
                add([_xy(q) for q in ring], True, comp_id, [None, None])
            continue
        s = starts[0]
        seq = walk_chain(adj, s, comp_set, endpoints)
        add([_xy(q) for q in seq], False, comp_id, [('e', seq[0]), ('e', seq[-1])])
    return edges


def prune_tips(edges):
    """简洁取向：删除中线长度不超过3像素、挂在分叉上的末梢。

    闭环、分叉间连线与独立线团不受影响；若某连通域会被整团删光则保留最长末梢。
    """
    def is_tip(e):
        if e['closed']:
            return False
        kinds = [n[0] if n else None for n in e['nodes']]
        return sorted(kinds) == ['e', 'j']

    candidates = [e for e in edges if is_tip(e) and _length(e['pts']) <= TIP_LIMIT]
    candidate_ids = {id(e) for e in candidates}
    keep = set()
    comps = {e['comp'] for e in edges}
    for comp in comps:
        if not any(id(e) in candidate_ids for e in edges if e['comp'] == comp):
            continue
        survivors = [e for e in edges if e['comp'] == comp and id(e) not in candidate_ids]
        if not survivors:
            longest = max((e for e in edges if e['comp'] == comp),
                          key=lambda e: _length(e['pts']))
            keep.add(id(longest))
    removed = [e for e in candidates if id(e) not in keep]
    removed_pts = [[[round(x, 1), round(y, 1)] for x, y in e['pts']] for e in removed]
    remaining = [e for e in edges if id(e) not in {id(e) for e in removed}]
    return remaining, removed_pts


def _seg_dist(p, a, b):
    px, py = p
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def rdp_open(points, eps):
    """迭代版 Ramer-Douglas-Peucker，避免长折线递归过深。"""
    if len(points) < 3:
        return points
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = points[i], points[j]
        worst, idx = 0.0, -1
        for k in range(i + 1, j):
            d = _seg_dist(points[k], a, b)
            if d > worst:
                worst, idx = d, k
        if idx >= 0 and worst > eps:
            keep[idx] = True
            stack.append((i, idx))
            stack.append((idx, j))
    return [p for p, k in zip(points, keep) if k]


def rdp(points, eps, closed):
    if not closed:
        return rdp_open(points, eps)
    if len(points) < 4:
        return points
    anchor = max(range(len(points)),
                 key=lambda i: math.hypot(points[i][0] - points[0][0],
                                          points[i][1] - points[0][1]))
    first = rdp_open(points[:anchor + 1], eps)
    second = rdp_open(points[anchor:] + [points[0]], eps)
    merged = first + second[1:-1]
    return merged if len(merged) >= 3 else points[:3]


def vectorize(a, threshold, mode):
    if mode not in ('faithful', 'simple'):
        raise Invalid('取向需为忠实或简洁')
    h, w = a.shape[:2]
    if max(h, w) > MAX_EDGE:
        raise Invalid('自动转单线仅支持最长边不超过256像素的底图')
    t = integer(threshold, 0, 255, '黑白阈值')
    mask = binarize(a, t)
    if not mask.any():
        return {'paths': [], 'removed_tips': [], 'ink_pixels': 0}
    skel = zhang_suen(mask)
    edges = graph_edges(skel, mask)
    removed = []
    if mode == 'simple':
        edges, removed = prune_tips(edges)
    eps = FAITHFUL_EPS if mode == 'faithful' else SIMPLE_EPS
    paths = []
    for e in edges:
        pts = rdp(e['pts'], eps, e['closed'])
        pts = [[round(x, 1), round(y, 1)] for x, y in pts]
        if e['closed']:
            if len(pts) < 3:
                continue
        elif len(pts) < 2:
            continue
        paths.append({'points': pts, 'closed': e['closed'], 'component': e['comp']})
    if len(paths) > MAX_PATHS:
        raise Invalid('识别路径超过500条上限，请调整黑白阈值或改用人工描线')
    for q in paths:
        if len(q['points']) > MAX_POINTS:
            raise Invalid('单条路径节点超过4096上限，请调整黑白阈值')
    return {'paths': paths, 'removed_tips': removed, 'ink_pixels': int(mask.sum())}
