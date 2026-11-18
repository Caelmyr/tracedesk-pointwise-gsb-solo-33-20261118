#!/usr/bin/env python3
"""走笔安排优化器的独立对拍测试：itertools 暴力枚举 vs planning.build_plan。"""
import glob
import itertools
import math
import os
import random
import sys

# -I 隔离模式下环境 PYTHONPATH 不生效，显式补入第三方包目录（避开工作目录注入）
for extra in glob.glob('/opt/*/site-packages'):
    if extra not in sys.path:
        sys.path.append(extra)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_lab.validation import Invalid  # noqa: E402
from tracedesk import planning  # noqa: E402

ORIGIN = (0.0, 0.0)


def brute_force(paths):
    """独立参考实现：全量枚举，不做任何剪枝，返回(最短距离, 挪动数)。"""
    n = len(paths)
    opts = []
    for q in paths:
        pts = q['points']
        if q.get('closed'):
            opts.append([(pts[k], pts[k - 1], k, k != 0) for k in range(len(pts))])
        else:
            opts.append([
                (pts[0], pts[-1], 0, False),
                (pts[-1], pts[0], len(pts) - 1, False),
            ])
    best = [math.inf, None]
    for perm in itertools.permutations(range(n)):
        for choice in itertools.product(*[range(len(opts[i])) for i in perm]):
            moved = sum(opts[i][k][3] for i, k in zip(perm, choice))
            if moved > 1:
                continue
            cur = ORIGIN
            total = 0.0
            for i, k in zip(perm, choice):
                start = opts[i][k][0]
                total += math.hypot(cur[0] - start[0], cur[1] - start[1])
                cur = opts[i][k][1]
            if total < best[0] - 1e-9 or (abs(total - best[0]) <= 1e-9 and
                                          (best[1] is None or moved < best[1])):
                best = [total, moved]
    return best[0], best[1]


def make_drawing(seed, n):
    rng = random.Random(seed)
    paths = []
    for i in range(n):
        closed = rng.random() < 0.4
        m = rng.randint(2, 10 if closed else 7)
        pts = [[round(rng.uniform(0, 200), 2), round(rng.uniform(0, 150), 2)]
               for _ in range(m)]
        paths.append({'id': f'p{i}', 'name': f'路径{i}', 'points': pts,
                      'closed': closed})
    return {'width': 256, 'height': 192, 'paths': paths,
            'stroke': '#256b72', 'line_width': 2, 'name': '测试稿'}


def test_optimality():
    for seed in range(300):
        n = 1 + seed % 6
        d = make_drawing(seed, n)
        plan = planning.build_plan(d, 0)
        ref_dist, ref_moved = brute_force(d['paths'])
        assert abs(plan['planned_travel'] - round(ref_dist, 2)) < 0.011, \
            f'seed {seed}: {plan["planned_travel"]} vs {ref_dist}'
        assert plan['moved_closed_count'] == ref_moved, \
            f'seed {seed}: moved {plan["moved_closed_count"]} vs {ref_moved}'
        # 空走分段之和应等于总距离（分段各自两位舍入，允许累积误差）
        seg_sum = round(sum(s['air_distance'] for s in plan['steps']), 2)
        assert abs(seg_sum - plan['planned_travel']) < 0.05 * n, (seed, seg_sum)
        # 每条路径恰好画一次
        assert sorted(s['path_id'] for s in plan['steps']) == \
            sorted(q['id'] for q in d['paths'])
        # 笔段首尾相接：空走从上个收笔点到下个起笔点
        for s, seg in zip(plan['steps'], plan['pen_up']):
            assert seg['to'] == s['start']
        for s, seg in zip(plan['steps'], plan['pen_up'][1:]):
            assert seg['from'] == s['end']
        assert plan['pen_up'][0]['from'] == [0.0, 0.0]
    print('test_optimality: 300 组随机图稿对拍通过')


def test_baseline():
    d = make_drawing(7, 4)
    paths = d['paths']
    expect = math.dist(ORIGIN, paths[0]['points'][0])
    for a, b in zip(paths, paths[1:]):
        end = a['points'][0] if a.get('closed') else a['points'][-1]
        expect += math.dist(end, b['points'][0])
    plan = planning.build_plan(d, 3)
    assert plan['revision'] == 3
    assert abs(plan['original_travel'] - round(expect, 2)) < 1e-9
    assert plan['saved'] == round(plan['original_travel'] - plan['planned_travel'], 2)
    print('test_baseline: 原空走距离（原顺序/原方向/原起点）正确')


def test_closed_direction_preserved():
    # 闭环：方案里的节点顺序必须是原循环顺序的旋转，不能反向
    d = make_drawing(11, 5)
    plan = planning.build_plan(d, 0)
    by_id = {q['id']: q for q in d['paths']}
    for s in plan['steps']:
        q = by_id[s['path_id']]
        if not q['closed']:
            continue
        ordered = planning.ordered_points(q, s)
        m = len(q['points'])
        k = s['start_index']
        assert ordered == [q['points'][(k + j) % m] for j in range(m)]
        assert s['reversed'] is False
    print('test_closed_direction_preserved: 闭环保持原循环方向')


def test_open_reverse():
    q = {'id': 'a', 'name': '开线', 'closed': False,
         'points': [[0, 0], [10, 10], [20, 0]]}
    d = {'width': 256, 'height': 192, 'paths': [q], 'stroke': '#000000',
         'line_width': 2, 'name': 't'}
    plan = planning.build_plan(d, 0)
    # 原点在(0,0)，正向起笔距离0，必选正向原端
    assert plan['steps'][0]['reversed'] is False
    q2 = dict(q, id='b', points=[[120, 120], [110, 110], [100, 100]])
    d['paths'] = [q2]
    plan = planning.build_plan(d, 0)
    # 原末端(100,100)离原点更近，应反向从末端起笔
    assert plan['steps'][0]['reversed'] is True
    assert plan['steps'][0]['start'] == [100.0, 100.0]
    assert plan['steps'][0]['end'] == [120.0, 120.0]
    print('test_open_reverse: 开线可从任一端起笔且选近端')


def test_tie_keeps_original():
    # 单条开线两端到原点等距 → 真正平局，应沿用原方向
    paths = [
        {'id': 'a', 'name': 'A', 'points': [[0, 10], [10, 0]], 'closed': False},
    ]
    d = {'width': 256, 'height': 192, 'paths': paths, 'stroke': '#000000',
         'line_width': 2, 'name': 't'}
    plan = planning.build_plan(d, 0)
    assert [s['path_id'] for s in plan['steps']] == ['a']
    assert plan['steps'][0]['reversed'] is False
    # 闭环等距时不挪起点
    cpts = [[10, 10], [20, 10], [20, 20], [10, 20]]
    paths2 = [{'id': 'c', 'name': 'C', 'points': cpts, 'closed': True}]
    d2 = dict(d, paths=paths2)
    plan2 = planning.build_plan(d2, 0)
    assert plan2['steps'][0]['start_index'] == 0
    assert plan2['moved_closed_count'] == 0
    print('test_tie_keeps_original: 等距优先原方向/原起点')


def test_single_closed_move_budget():
    # 三个闭环，离原点最近的点都不是节点0；全局最优也只允许挪一个
    rng = random.Random(99)
    paths = []
    for i in range(3):
        m = 5
        pts = [[round(50 + rng.uniform(0, 120), 2),
                round(40 + rng.uniform(0, 100), 2)] for _ in range(m)]
        paths.append({'id': f'c{i}', 'name': f'环{i}', 'points': pts,
                      'closed': True})
    d = {'width': 256, 'height': 192, 'paths': paths, 'stroke': '#000000',
         'line_width': 2, 'name': 't'}
    plan = planning.build_plan(d, 0)
    assert plan['moved_closed_count'] <= 1
    ref_dist, ref_moved = brute_force(paths)
    assert abs(plan['planned_travel'] - round(ref_dist, 2)) < 0.011
    print('test_single_closed_move_budget: 至多挪动一个闭环起点且仍最优')


def test_limits():
    d = make_drawing(1, 6)
    planning.build_plan(d, 0)  # 6 条可以
    d7 = make_drawing(2, 7)
    try:
        planning.build_plan(d7, 0)
        raise AssertionError('7 条路径应被拒绝')
    except Invalid as e:
        assert '6' in str(e)
    big = {'id': 'x', 'name': '大环', 'closed': True,
           'points': [[i, i] for i in range(11)]}
    d_big = dict(make_drawing(3, 1), paths=[big])
    try:
        planning.build_plan(d_big, 0)
        raise AssertionError('11 节点闭环应被拒绝')
    except Invalid as e:
        assert '10' in str(e)
    # 开线不受 10 节点限制
    open_big = dict(big, id='y', closed=False)
    planning.build_plan(dict(d_big, paths=[open_big]), 0)
    # 空稿
    try:
        planning.build_plan(dict(d_big, paths=[]), 0)
        raise AssertionError('空稿应被拒绝')
    except Invalid:
        pass
    print('test_limits: 6 路径/闭环10节点/空稿限制正确')


def test_svg_output():
    import re
    d = make_drawing(5, 5)
    plan = planning.build_plan(d, 0)
    svg = planning.plan_svg(d, plan).decode()
    assert svg.count('<path') == 5
    ids = [s['path_id'] for s in plan['steps']]
    pos = [svg.index(f'data-path-id="{i}"') for i in ids]
    assert pos == sorted(pos), 'SVG 路径顺序必须与预览顺序一致'
    closed_total = sum(1 for q in d['paths'] if q['closed'])
    assert svg.count(' Z') == closed_total
    tags = re.findall(r'<path d="([^"]+)" data-path-id="([^"]+)"[^>]*/>', svg)
    assert [pid for _, pid in tags] == ids
    for dattr, pid in tags:
        s = next(x for x in plan['steps'] if x['path_id'] == pid)
        q = next(q for q in d['paths'] if q['id'] == pid)
        ordered = planning.ordered_points(q, s)
        head = f'M {ordered[0][0]:.4f} {ordered[0][1]:.4f}'
        assert dattr.startswith(head), (dattr, head)
        # 节点集合与闭合关系不变
        assert len(ordered) == len(q['points'])
        assert {tuple(v) for v in ordered} == {tuple(v) for v in q['points']}
        if q['closed']:
            assert dattr.endswith(' Z')
    print('test_svg_output: SVG 顺序/方向/起点/闭合与方案一致')


if __name__ == '__main__':
    test_optimality()
    test_baseline()
    test_closed_direction_preserved()
    test_open_reverse()
    test_tie_keeps_original()
    test_single_closed_move_budget()
    test_limits()
    test_svg_output()
    print('全部通过')
