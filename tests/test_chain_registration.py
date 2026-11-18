"""串接配准（chain alignment）验证脚本：不依赖 pytest，直接 python3 -I 运行。"""
import sys
import numpy as np

sys.path.insert(0, '/workspace')

from image_lab.registration import align_set
from image_lab import validation as v


def textured(height, width, seed=7):
    rng = np.random.default_rng(seed)
    base = rng.random((height, width, 3), dtype=np.float32) * .35
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base[..., 0] += .25 * np.sin(xx / 7.0) + .15 * np.cos(yy / 5.0)
    base[..., 1] += .2 * np.cos((xx + yy) / 11.0)
    base[..., 2] += .18 * np.sin(yy / 6.5)
    for cx, cy in [(40, 55), (160, 90), (95, 175), (210, 200)]:
        disk = ((xx - cx) ** 2 + (yy - cy) ** 2) < 60
        base[disk] += rng.random(3) * .4
    return np.clip(base, 0, 1).astype(np.float32)


def crop_scene(scene, dx, dy, h, w):
    return scene[dy:dy + h, dx:dx + w].copy()


def prepared(images, names):
    frames = [{'id': '%032d' % i, 'name': name, 'asset': 'x' * 64} for i, name in enumerate(names)]
    return {'images': images, 'frames': frames, 'reference_index': 0, 'size': (images[0].shape[1], images[0].shape[0])}


def settings(**overrides):
    data = {'method': 'phase', 'motion': 'translation', 'minimum_overlap': .25, 'minimum_correlation': .6, 'chaining': True}
    data.update(overrides)
    return v.alignment_settings(data)


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print('  ok -', message)


h, w = 256, 256
scene = textured(h + 260, w + 260)
# 相机平移：frame_i 在场景中的裁切原点 (dx_i, dy_i)，位移约 95~110 像素
origins = [(0, 0), (96, 2), (205, 5)]
images = [crop_scene(scene, dx, dy, h, w) for dx, dy in origins]
prep = prepared(images, ['ref', 'mid', 'far'])

print('场景1：远帧与参考重叠不足，但可经中间帧接通')
report = align_set(prep, settings())
check(report['chaining'] is True and report['applicable'] is True, '报告标记为串接且可应用')
check(report['disconnected_frames'] == [], '没有接不起来的照片')
pairs = {(p['a'], p['b']): p for p in report['pairs']}
check(not pairs[(0, 2)]['trusted'] and pairs[(0, 2)]['reason'] is not None, '参考-远帧对不可信并记录原因')
check(pairs[(0, 1)]['trusted'] and pairs[(1, 2)]['trusted'], '相邻对均可信')
# 相机移动 (dx,dy) 时，warp 矩阵把移动帧拉回参考坐标的平移量恰为 (+dx, +dy)（与 phase_align 既有约定一致）
expected = [(96, 2), (205, 5)]
for index, (ex, ey) in enumerate(expected, start=1):
    shift = report['frames'][index]['details']['shift']
    check(abs(shift[0] - ex) < 1.5 and abs(shift[1] - ey) < 1.5,
          f'{prep["frames"][index]["name"]} 相对参考位移 {shift} 接近期望 ({ex},{ey})')
    check(report['frames'][index]['details']['connected'] is True, '帧标记为已接通')
check(report['frames'][0]['details']['shift'] == [0., 0.], '参考帧固定不动')
for pair in report['pairs']:
    if pair['trusted']:
        check(pair['residual']['max_abs'] < 1.0, f'可信对 {pair["a_name"]}<->{pair["b_name"]} 剩余偏差 {pair["residual"]} 小于1像素')
check(all(idx in report['frames'][2]['details']['used_pairs'] for idx in report['frames'][1]['details']['used_pairs'])
      or len(report['frames'][2]['details']['used_pairs']) >= 1, '远帧记录了使用的可信配对索引')
# 位移必须相对参考而非相对前一张
far_chain_relative = report['frames'][2]['details']['shift']
check(abs(far_chain_relative[0] - 205) < 1.5, '远帧报告的是相对参考帧的累计位移而非相对中间帧')

print('场景2：阈值过高导致链断裂，任务仍出报告但不能应用')
strict = settings(minimum_overlap=.9)
report2 = align_set(prep, strict)
check(report2['applicable'] is False, '存在接不起来的照片时 applicable=False')
check(set(report2['disconnected_frames']) == {'mid', 'far'}, '点名接不起来的照片: ' + str(report2['disconnected_frames']))
check(not report2['frames'][2]['details']['connected'] and report2['frames'][2]['quality'] is None, '未接通帧无质量数据')
check(report2['frames'][0]['details']['connected'] is True, '参考帧始终接通')
check(all(p['trusted'] is False for p in report2['pairs']), '所有配对均未达到下限')

print('场景3：多条可信路径同时参与，minimax 让最差方向也对齐')
# 五帧序列，制造一个可经不同中间帧接通的帧；加轻微噪声
rng = np.random.default_rng(11)
origins5 = [(0, 0), (70, 1), (140, 3), (210, 4), (105, 60)]
images5 = [np.clip(crop_scene(scene, dx, dy, h, w) + rng.normal(0, .01, (h, w, 3)), 0, 1).astype(np.float32) for dx, dy in origins5]
prep5 = prepared(images5, ['r', 'a', 'b', 'c', 'loop'])
report5 = align_set(prep5, settings(minimum_overlap=.12, minimum_correlation=.5))
check(report5['applicable'] is True, '五帧全部接通: ' + str(report5['disconnected_frames']))
worst = max(p['residual']['max_abs'] for p in report5['pairs'] if p['trusted'])
check(worst < 1.5, f'所有可信对的最差剩余偏差 {worst:.3f} 像素被全局求解压住')
trusted_count = sum(1 for p in report5['pairs'] if p['trusted'])
check(trusted_count >= 4, f'纳入的可信对数量 {trusted_count} 至少覆盖每条相邻链接')
loop_shift = report5['frames'][4]['details']['shift']
check(abs(loop_shift[0] - 105) < 2 and abs(loop_shift[1] - 60) < 2, f'loop 帧位移 {loop_shift} 相对参考正确')

print('场景4：使用范围限制')
for bad, msg in [
    (prepared(images[:2], ['a', 'b']), '少于3张应拒绝'),
]:
    try:
        align_set(bad, settings())
        raise AssertionError(msg)
    except v.Invalid as error:
        check('3到5张' in str(error), msg + f'（{error}）')
big = prepared([np.zeros((300, 400, 3), np.float32) for _ in range(3)], ['a', 'b', 'c'])
try:
    align_set(big, settings())
    raise AssertionError('最长边超过384应拒绝')
except v.Invalid as error:
    check('384' in str(error), f'最长边限制生效（{error}）')
mismatch = prepared([np.zeros((256, 256, 3), np.float32), np.zeros((256, 256, 3), np.float32), np.zeros((200, 256, 3), np.float32)], ['a', 'b', 'c'])
try:
    align_set(mismatch, settings())
    raise AssertionError('尺寸不同应拒绝')
except v.Invalid as error:
    check('尺寸相同' in str(error), f'同尺寸限制生效（{error}）')
six = prepared([np.zeros((256, 256, 3), np.float32) for _ in range(6)], [str(i) for i in range(6)])
try:
    align_set(six, settings())
    raise AssertionError('超过5张应拒绝')
except v.Invalid as error:
    check('3到5张' in str(error), f'张数上限生效（{error}）')

print('场景5：关闭串接时保持原有单对配准行为')
plain = v.alignment_settings({'method': 'phase', 'chaining': False})
report_plain = align_set(prep, plain)
check('chaining' not in report_plain, '普通配准报告不含串接字段')
check(len(report_plain['frames']) == 3 and report_plain['frames'][0]['details']['method'] == 'reference', '普通配准逐帧相对参考输出')

print('\n全部串接配准检查通过')
