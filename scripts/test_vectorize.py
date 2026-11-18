#!/usr/bin/env python3
"""自动转单线（vectorize）行为验证。运行：python3 -I scripts/test_vectorize.py"""
import math
import os
import sys
import tempfile

_ENV_SITE = '/opt/pointwise-env/site-packages'
if os.path.isdir(_ENV_SITE) and _ENV_SITE not in sys.path:
    sys.path.insert(0, _ENV_SITE)

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from image_lab.validation import Invalid
from tracedesk.vectorize import (vectorize, binarize, zhang_suen, graph_edges,
                                 prune_tips, _length, TIP_LIMIT)


def canvas(w=256, h=192):
    return np.array(Image.new('RGBA', (w, h), 'white'))


def line(a, pts, width=7, fill=(20, 20, 20, 255)):
    im = Image.fromarray(a, 'RGBA')
    ImageDraw.Draw(im).line(pts, fill=fill, width=width, joint='curve')
    a[:] = np.array(im)
    return a


def ellipse(a, box, width=5, fill=(20, 20, 20, 255)):
    im = Image.fromarray(a, 'RGBA')
    ImageDraw.Draw(im).ellipse(box, outline=fill, width=width)
    a[:] = np.array(im)
    return a


def closed_count(r):
    return sum(q['closed'] for q in r['paths'])


def open_count(r):
    return sum(not q['closed'] for q in r['paths'])


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print('ok -', msg)


# 1. 单根粗直线只产生一条路径，且节点在笔画中部而非两侧轮廓
a = canvas(80, 80)
line(a, [(10, 40), (70, 40)], width=11)
r = vectorize(a, 128, 'faithful')
check(len(r['paths']) == 1, f'粗直线只出一条路径（得到{len(r["paths"])}条）')
q = r['paths'][0]
check(not q['closed'], '粗直线是开口路径')
check(all(abs(y - 40) <= 1.5 for x, y in q['points']), '中线位于笔画中部（非两侧轮廓）')
check(q['points'][0][0] <= 20 and q['points'][-1][0] >= 60, '中线贯穿笔画两端（允许圆角笔端自然回缩）')

# 2. 真闭环：椭圆描边得到闭合路径
a = canvas(120, 120)
ellipse(a, (20, 20, 100, 100), width=7)
r = vectorize(a, 128, 'faithful')
check(closed_count(r) == 1 and len(r['paths']) == 1, f'椭圆得到1条闭环（{r["paths"] and closed_count(r)}闭）')

# 3. 分叉：Y 形交叉拆成多条在同一点相接的路径
a = canvas(120, 120)
line(a, [(60, 100), (60, 60)], width=7)
line(a, [(60, 60), (20, 20)], width=7)
line(a, [(60, 60), (100, 20)], width=7)
r = vectorize(a, 128, 'faithful')
check(len(r['paths']) >= 3, f'Y形至少拆成3条相接路径（得到{len(r["paths"])}）')
# 所有分叉边应共享同一节点坐标
shared = [tuple(q['points'][0]) for q in r['paths']]
check(len(set(shared)) == 1, f'分叉路径共享同一节点（{set(shared)}）')

# 3b. 十字交叉拆成4条在同一节点相接的路径
a = canvas(120, 120)
line(a, [(20, 60), (100, 60)], 7)
line(a, [(60, 20), (60, 100)], 7)
r = vectorize(a, 128, 'faithful')
check(len(r['paths']) == 4, f'十字交叉拆成4条路径（得到{len(r["paths"])}）')
check(len({tuple(q['points'][0]) for q in r['paths']}) == 1, '十字的4条路径共享同一节点')

# 3c. 主干上两个分叉形成两个不同的共享节点
a = canvas(200, 120)
line(a, [(20, 60), (180, 60)], 7)
line(a, [(60, 60), (60, 20)], 5)
line(a, [(120, 60), (120, 100)], 5)
r = vectorize(a, 128, 'faithful')
check(len(r['paths']) >= 4, f'双分叉网络至少4条路径（得到{len(r["paths"])}）')
check(len({tuple(q['points'][0]) for q in r['paths']}) == 2, '两个分叉是两个不同节点')
check(not any(q['closed'] for q in r['paths']), '双分叉网络无伪闭环')

# 4. 彼此分开的线团不能被连起来：两个独立线团 -> 两个 component
a = canvas(200, 100)
line(a, [(15, 50), (55, 50)], width=7)
line(a, [(120, 50), (180, 50)], width=7)
r = vectorize(a, 128, 'faithful')
comps = {q['component'] for q in r['paths']}
check(len(comps) == 2, f'两个独立线团保持分离（得到{len(comps)}个连通域）')

# 5. 简洁模式删除 <=3px 末梢，忠实模式保留
a = canvas(160, 120)
line(a, [(80, 100), (80, 50)], width=7)          # 主干
line(a, [(80, 50), (30, 50)], width=7)           # 长分支
line(a, [(80, 50), (120, 20)], width=7)          # 另一长分支
line(a, [(80, 50), (92, 50)], width=3)           # 较短开口末梢
rf = vectorize(a, 128, 'faithful')
rs = vectorize(a, 128, 'simple')
short_open = [q for q in rf['paths'] if not q['closed'] and _length(q['points']) <= TIP_LIMIT]
long_open = [q for q in rf['paths'] if not q['closed'] and _length(q['points']) > TIP_LIMIT]
if short_open:
    check(len(rs['paths']) < len(rf['paths']), f'简洁模式路径更少（忠实{len(rf["paths"])} -> 简洁{len(rs["paths"])}）')
    check(len(rs['removed_tips']) >= 1, '简洁模式报告被删除的末梢')
    check(all(_length(t) <= TIP_LIMIT for t in rs['removed_tips']), '被删末梢中线长度均不超过3像素')
    check(len(rs['paths']) >= len(long_open), '简洁模式保留所有长分支')
else:
    print('ok - 该合成图无<=3px开口末梢，改用剪枝单元测试覆盖')
# 小闭环不能被当末梢删掉
a = canvas(120, 120)
ellipse(a, (20, 20, 60, 60), width=5)            # 小闭环
ellipse(a, (70, 70, 110, 110), width=5)          # 另一小闭环
rs = vectorize(a, 128, 'simple')
check(closed_count(rs) == 2, '简洁模式保留全部小闭环')
check(open_count(rs) == 0, '小闭环未被误判为短末梢')

# 6. 独立线团不能被整团删掉（整团都是短末梢时保留最长者）
a = canvas(80, 80)
line(a, [(15, 40), (26, 40)], width=3)           # 短独立线团
r = vectorize(a, 128, 'simple')
check(len(r['paths']) >= 1, '独立短线团在简洁模式下仍保留')

# 7. 简洁结果相对忠实结果偏移不超过2px（逐点到忠实折线距离）
def seg_dist(p, a_, b_):
    dx, dy = b_[0] - a_[0], b_[1] - a_[1]
    if dx == 0 and dy == 0:
        return math.hypot(p[0] - a_[0], p[1] - a_[1])
    t = max(0.0, min(1.0, ((p[0] - a_[0]) * dx + (p[1] - a_[1]) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - (a_[0] + t * dx), p[1] - (a_[1] + t * dy))


a = canvas(256, 192)
line(a, [(20, 150), (80, 80), (130, 140), (180, 50), (235, 150)], width=9)
line(a, [(80, 80), (85, 25)], width=5)
ellipse(a, (150, 120, 200, 165), width=7)  # 与折线留出间隙，形成独立闭环线团
rf = vectorize(a, 128, 'faithful')
rs = vectorize(a, 128, 'simple')
check(len({q['component'] for q in rs['paths']}) >= 2, '折线与闭环是两个独立线团')
check(closed_count(rs) >= 1 and closed_count(rs) == closed_count(rf), '简洁模式保留全部闭环（含小闭环）')
worst = 0.0
for qs in rs['paths']:
    refs = [q for q in rf['paths'] if q['component'] == qs['component']]
    best_ref = 1e9
    for ref in refs:
        poly = ref['points'] + ([ref['points'][0]] if ref['closed'] else [])
        d = max(min(seg_dist(pt, poly[k], poly[k + 1]) for k in range(len(poly) - 1))
                for pt in qs['points'])
        best_ref = min(best_ref, d)
    worst = max(worst, best_ref)
check(worst <= 2.0, f'简洁相对忠实偏移<=2px（实测{worst:.2f}px）')
check(sum(len(q['points']) for q in rs['paths']) <=
      sum(len(q['points']) for q in rf['paths']), '简洁模式节点数不增加')

# 8. 阈值调节改变笔画范围
a = canvas(80, 80)
line(a, [(10, 40), (70, 40)], width=7, fill=(110, 110, 110, 255))  # 灰色笔画
r_dark = vectorize(a, 90, 'faithful')
r_light = vectorize(a, 200, 'faithful')
check(r_dark['ink_pixels'] == 0, '高阈值（只取更暗像素）下灰线被排除')
check(r_light['ink_pixels'] > 0 and len(r_light['paths']) == 1, '低阈值下灰线被识别为笔画')

# 9. 透明背景与不透明线稿
a = np.zeros((60, 60, 4), dtype=np.uint8)  # 全透明
im = Image.fromarray(a, 'RGBA')
ImageDraw.Draw(im).line([(10, 30), (50, 30)], fill=(0, 0, 0, 255), width=5)
a[:] = np.array(im)
r = vectorize(a, 128, 'faithful')
check(len(r['paths']) == 1, '透明背景上的不透明线稿正常识别')

# 10. 最长边超过256被拒绝
a = canvas(257, 100)
try:
    vectorize(a, 128, 'faithful')
    raise AssertionError('应拒绝257px底图')
except Invalid as e:
    check('256' in str(e), f'超尺寸底图被拒绝（{e}）')

# 11. 空白底图返回空结果而非报错
r = vectorize(canvas(), 128, 'faithful')
check(r['paths'] == [], '全白底图返回空路径集')

# 12. 非法模式与阈值
for bad in (None, '', 'magic', 3):
    try:
        vectorize(canvas(), 128, bad)
        raise AssertionError('应拒绝非法模式')
    except Invalid:
        pass
for bad in (-1, 256, 1.5):
    try:
        vectorize(canvas(), bad, 'faithful')
        raise AssertionError('应拒绝非法阈值')
    except Invalid:
        pass
print('ok - 非法模式/阈值均被拒绝')

# 13. 细化是单像素宽，且骨架像素全部落在笔画内部
a = canvas(200, 120)
line(a, [(20, 60), (180, 60)], width=13)
mask = binarize(a, 128)
skel = zhang_suen(mask)
check(skel.sum() > 0, '骨架非空')
check(bool((skel & ~mask).sum() == 0), '骨架完全位于笔画掩膜内')
# 单像素宽：骨架上不存在2x2全为骨架的方块
h, w = skel.shape
blocks = skel[:-1, :-1] & skel[:-1, 1:] & skel[1:, :-1] & skel[1:, 1:]
check(not blocks.any(), '骨架无2x2实心块（单像素宽）')

# 14. 剪枝函数在含闭环+短末梢的混合线团中只删末梢不删环
a = canvas(160, 160)
ellipse(a, (30, 30, 110, 110), width=6)
line(a, [(70, 110), (70, 114)], width=3)  # 环上的短毛刺
r = vectorize(a, 128, 'simple')
check(closed_count(r) == 1, '带毛刺的闭环在简洁模式下仍闭合')

# 15. 剪枝器单元测试：短末梢删除、闭环/连线/独立线团保护
def edge(pts, closed=False, comp=1, nodes=None):
    if nodes is None:
        nodes = [None, None] if closed else [('e', (0, 0)), ('j', 0)]
    return {'pts': pts, 'closed': closed, 'comp': comp, 'nodes': nodes}


tip = edge([[0, 0], [2, 0]])                       # 长2的分叉末梢
long_arm = edge([[0, 0], [50, 0]], nodes=[('e', (50, 0)), ('j', 0)])
loop = edge([[0, 0], [1, 0], [1, 1]], closed=True, nodes=[('j', 0), ('j', 0)])
bridge = edge([[0, 0], [1, 0]], nodes=[('j', 0), ('j', 1)])
remaining, removed = prune_tips([tip, long_arm, loop, bridge])
check(len(remaining) == 3 and len(removed) == 1, '只删除短末梢，保留长分支/闭环/分叉连线')
# 独立短线团（两条末梢组成，整团删光时保留最长者）
t1 = edge([[0, 0], [1, 0]], comp=2, nodes=[('e', (0, 0)), ('j', 2)])
t2 = edge([[0, 0], [2, 0]], comp=2, nodes=[('e', (2, 0)), ('j', 2)])
remaining, removed = prune_tips([t1, t2])
check(len(remaining) == 1 and len(removed) == 1, '独立线团不会被整团删除（保留最长末梢）')
# 纯闭环永不进候选
remaining, removed = prune_tips([edge([[0, 0], [1, 0], [1, 1]], closed=True, comp=3)])
check(len(remaining) == 1 and removed == [], '纯闭环在简洁模式下完整保留')

print('\n全部通过。')
