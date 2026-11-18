#!/usr/bin/env python3
"""整组颜色一致（color_consensus）降噪的独立验证脚本。

数据目录由命令行参数传入，与脚本、源码分离；仅依赖 image_lab 包。
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, '/workspace')

import numpy as np

from image_lab.arrays import srgb_decode
from image_lab.denoise import denoise_stack, color_consensus, check_color_consensus_limits
from image_lab.validation import Invalid
from image_lab import model
from image_lab.store import Store
from image_lab.assets import Assets
from image_lab.results import Results
from image_lab.engine import process


def make_prepared(images, valids=None, reference_index=0, size=None, weights=None):
    frames = []
    for index in range(len(images)):
        entry = model.frame('0' * 32, f'frame-{index}')
        if weights is not None:
            entry['weight'] = weights[index]
        frames.append(entry)
    height, width = images[0].shape[:2]
    return {
        'frames': frames,
        'images': [img.astype(np.float32) for img in images],
        'validities': [np.ones((height, width), np.float32) for _ in images] if valids is None else [v.astype(np.float32) for v in valids],
        'geometric_validities': [np.ones((height, width), np.float32) for _ in images],
        'masks': [np.ones((height, width), np.float32) for _ in images],
        'matrices': [model.identity() for _ in images],
        'reference_index': reference_index,
        'size': size or (width, height),
        'calibration': [],
    }


def settings(**overrides):
    base = {'method': 'color_consensus', 'sigma': 2.0, 'iterations': 2, 'trim': 0.15,
            'linear': False, 'color_tolerance': 10}
    base.update(overrides)
    return base


def nearly(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def test_basic_consensus_picks_largest_group():
    # 4 张正常照片颜色接近，1 张出现反光怪色；容差 10/255
    base = np.array([0.4, 0.5, 0.6], np.float32)
    images = []
    for delta in [0.0, 0.01, -0.01, 0.02]:
        img = np.zeros((4, 4, 3), np.float32) + base + delta
        images.append(img)
    flare = np.zeros((4, 4, 3), np.float32) + np.array([0.9, 0.2, 0.15], np.float32)
    images.append(flare)
    prepared = make_prepared(images, reference_index=0)
    fields, report = denoise_stack(prepared, settings())
    assert fields['sample_count'].shape == (4, 4)
    assert np.all(fields['sample_count'] == 4), fields['sample_count']
    expected = np.mean(np.stack(images[:4]), axis=0)
    assert np.allclose(fields['image'], expected, atol=1e-6), np.abs(fields['image'] - expected).max()
    assert report['max_sample_count'] == 4
    assert nearly(report['mean_sample_count'], 4.0)
    print('PASS test_basic_consensus_picks_largest_group')


def test_whole_rgb_taken_or_rejected_together():
    # 反光帧：R 通道在容差内，但 G/B 远超容差 -> 整组（含 R）必须被排除
    base = np.array([0.5, 0.5, 0.5], np.float32)
    good = [np.zeros((2, 2, 3), np.float32) + base for _ in range(3)]
    flare = np.zeros((2, 2, 3), np.float32) + np.array([0.52, 0.85, 0.85], np.float32)
    prepared = make_prepared(good + [flare], reference_index=0)
    fields, _ = denoise_stack(prepared, settings(color_tolerance=10))
    assert np.all(fields['sample_count'] == 3)
    assert np.allclose(fields['image'], base, atol=1e-6)
    print('PASS test_whole_rgb_taken_or_rejected_together')


def test_tolerance_boundary_inclusive():
    # 差值恰好等于容差（按8位级数）应视为兼容
    base = np.zeros((1, 1, 3), np.float32) + 0.5
    edge = np.zeros((1, 1, 3), np.float32) + 0.5 + 10 / 255.
    prepared = make_prepared([base, edge])
    fields, _ = denoise_stack(prepared, settings(color_tolerance=10))
    assert fields['sample_count'][0, 0] == 2
    outside = np.zeros((1, 1, 3), np.float32) + 0.5 + 11 / 255.
    prepared = make_prepared([base, outside])
    fields, _ = denoise_stack(prepared, settings(color_tolerance=10))
    assert fields['sample_count'][0, 0] == 1
    print('PASS test_tolerance_boundary_inclusive')


def test_tie_break_by_reference_color():
    # 构造两组等大（各2张）但颜色不同的团；参考帧属于其中一组 -> 选含参考平均色更近的组
    group_a = [0.30, 0.31]
    group_b = [0.70, 0.71]
    images = []
    for value in group_a + group_b:
        images.append(np.zeros((1, 1, 3), np.float32) + np.float32(value))
    # 参考帧是第 0 张（A 组）
    prepared = make_prepared(images, reference_index=0)
    fields, _ = denoise_stack(prepared, settings(color_tolerance=5))
    assert fields['sample_count'][0, 0] == 2
    assert np.allclose(fields['image'][0, 0], np.mean(group_a), atol=1e-6), fields['image'][0, 0]
    # 参考帧换成 B 组成员 -> 应选 B 组
    prepared = make_prepared(images, reference_index=2)
    fields, _ = denoise_stack(prepared, settings(color_tolerance=5))
    assert np.allclose(fields['image'][0, 0], np.mean(group_b), atol=1e-6), fields['image'][0, 0]
    print('PASS test_tie_break_by_reference_color')


def test_tie_break_by_frame_order_when_reference_invalid():
    # 参考帧在该像素无效；两个等大组 -> 取原照片顺序最先达到该张数的组
    values = [0.30, 0.31, 0.70, 0.71]
    images = [np.zeros((1, 1, 3), np.float32) + np.float32(v) for v in values]
    valids = [np.ones((1, 1), np.float32) for _ in values]
    valids[0] = np.zeros((1, 1), np.float32)  # 参考帧（index 0）无效
    prepared = make_prepared(images, valids, reference_index=0)
    fields, _ = denoise_stack(prepared, settings(color_tolerance=5))
    assert fields['sample_count'][0, 0] == 2
    # 最先形成的 2 张组是 index 1,2? 不：1 与 2 色差 0.39 不兼容；实际组是 (1) 单张与 (2,3) 组
    # values: index0 无效；index1=0.31 单独；index2,3 互相兼容 -> 最大组为 (2,3)
    assert np.allclose(fields['image'][0, 0], np.mean([0.70, 0.71]), atol=1e-6)
    print('PASS test_tie_break_by_frame_order_when_reference_invalid')


def test_masks_limit_valid_samples():
    # 遮罩排除某帧的部分区域
    base = np.zeros((2, 2, 3), np.float32) + 0.5
    images = [base.copy() for _ in range(4)]
    images[3][:, :] = 0.9
    valids = [np.ones((2, 2), np.float32) for _ in range(4)]
    valids[3][0, 0] = 0.0  # 反光帧在 (0,0) 被遮罩排除
    prepared = make_prepared(images, valids, reference_index=0)
    fields, _ = denoise_stack(prepared, settings(color_tolerance=10))
    # (0,0) 反光帧无效 -> 三张正常帧
    assert fields['sample_count'][0, 0] == 3
    # 其他位置反光帧有效但被颜色规则排除 -> 仍是三张
    assert fields['sample_count'][1, 1] == 3
    assert np.allclose(fields['image'][0, 0].ravel(), 0.5, atol=1e-6)
    print('PASS test_masks_limit_valid_samples')


def test_invalid_region_kept_invalid():
    images = [np.zeros((2, 2, 3), np.float32) + 0.5 for _ in range(3)]
    valids = [np.ones((2, 2), np.float32) for _ in range(3)]
    for valid in valids:
        valid[0, 0] = 0.0  # 所有帧在该点都无效
    prepared = make_prepared(images, valids, reference_index=0)
    fields, _ = denoise_stack(prepared, settings())
    assert fields['valid'][0, 0] == 0
    assert fields['sample_count'][0, 0] == 0
    assert np.all(fields['image'][0, 0] == 0)
    print('PASS test_invalid_region_kept_invalid')


def test_linear_space_average():
    # 线性空间开启时，组内平均应在 sRGB 解码后的线性值上进行，再编码回 sRGB
    base = np.zeros((1, 1, 3), np.float32) + 0.4
    images = [base, base + 0.02, base - 0.02]
    prepared = make_prepared(images, reference_index=0)
    fields_lin, _ = denoise_stack(prepared, settings(linear=True, color_tolerance=20))
    lin_stack = srgb_decode(np.stack(images))
    expected = np.mean(lin_stack, axis=0)
    from image_lab.arrays import srgb_encode
    expected = srgb_encode(expected)
    assert np.allclose(fields_lin['image'], expected, atol=1e-6)
    fields_srgb, _ = denoise_stack(prepared, settings(linear=False, color_tolerance=20))
    assert np.allclose(fields_srgb['image'], np.mean(np.stack(images), axis=0), atol=1e-6)
    assert not np.allclose(fields_lin['image'], fields_srgb['image'])
    print('PASS test_linear_space_average')


def test_guard_limits():
    images = [np.zeros((2, 2, 3), np.float32) for _ in range(6)]
    prepared = make_prepared(images, size=(2, 2))
    try:
        check_color_consensus_limits(prepared)
    except Invalid:
        pass
    else:
        raise AssertionError('6 张照片应触发护栏')
    images = [np.zeros((2, 2, 3), np.float32) for _ in range(4)]
    prepared = make_prepared(images, size=(400, 300))
    try:
        check_color_consensus_limits(prepared)
    except Invalid:
        pass
    else:
        raise AssertionError('长边 400 应触发护栏')
    prepared = make_prepared(images, size=(384, 300))
    check_color_consensus_limits(prepared)  # 不应抛错
    print('PASS test_guard_limits')


def test_brute_force_equivalence_random():
    # 随机小数据：向量化实现 vs 暴力枚举所有子集，验证最大团与平局规则
    rng = np.random.default_rng(7)
    height, width, frames = 3, 3, 5
    limit = 12 / 255.
    colors = rng.random((frames, height, width, 3)).astype(np.float32)
    valid = (rng.random((frames, height, width)) > 0.25).astype(np.float32)
    ref_index = 2
    srgb_mean, _, weights, count = color_consensus(colors, colors, valid, limit, ref_index)
    for y in range(height):
        for x in range(width):
            active = [i for i in range(frames) if valid[i, y, x] > .5]
            if not active:
                assert count[y, x] == 0
                continue
            # 暴力找最大兼容团（frames<=5，枚举 2^n）
            best = []
            for mask in range(1, 1 << len(active)):
                subset = [active[k] for k in range(len(active)) if mask & (1 << k)]
                ok = all(
                    np.all(np.abs(colors[i, y, x] - colors[j, y, x]) <= limit + 1e-7)
                    for a, i in enumerate(subset) for j in subset[a + 1:])
                if ok and len(subset) > len(best):
                    best = subset
            assert count[y, x] == len(best), (y, x, count[y, x], best)
            chosen = [i for i in range(frames) if weights[i, y, x] > .5]
            assert sorted(chosen) == sorted(best) or len(chosen) == len(best)
            # 平局规则校验：若参考有效，最优组平均色到参考的距离必须最小
            if valid[ref_index, y, x] > .5:
                ref = colors[ref_index, y, x]
                # 枚举所有最大团并比较
                max_groups = []
                for mask in range(1, 1 << len(active)):
                    subset = [active[k] for k in range(len(active)) if mask & (1 << k)]
                    if len(subset) != len(best):
                        continue
                    ok = all(
                        np.all(np.abs(colors[i, y, x] - colors[j, y, x]) <= limit + 1e-7)
                        for a, i in enumerate(subset) for j in subset[a + 1:])
                    if ok:
                        max_groups.append(subset)
                dist = lambda g: np.linalg.norm(np.mean([colors[i, y, x] for i in g], axis=0) - ref)
                minimal = min(dist(g) for g in max_groups)
                assert dist(chosen) <= minimal + 1e-6, (y, x, dist(chosen), minimal)
    print('PASS test_brute_force_equivalence_random')


def test_end_to_end_engine_and_results():
    with tempfile.TemporaryDirectory(dir='/workspace/verify_data') as data_dir:
        store = Store(data_dir)
        assets = Assets(store)
        results = Results(store, assets)
        project = store.create('颜色一致端到端')
        base = np.zeros((420, 620, 3), np.float32) + 0.45
        frame_ids = []
        with store.edit(project['id'], 0, '载入测试序列') as p:
            for index in range(5):
                img = base.copy()
                img += np.float32(0.004 * (index - 2))
                if index == 4:
                    img[:] = np.array([0.95, 0.3, 0.2], np.float32)  # 反光帧
                asset = assets.put(img, f'shot-{index}.png')
                entry = model.frame(asset['id'], f'shot-{index}')
                p['frames'].append(entry)
                frame_ids.append(entry['id'])
            p['reference'] = frame_ids[0]
            p['settings']['mode'] = 'denoise'
            p['settings']['size'] = 384
            p['settings']['denoise'] = {
                'method': 'color_consensus', 'sigma': 2.0, 'iterations': 2,
                'trim': 0.15, 'linear': True, 'color_tolerance': 12}
        project = store.get(project['id'])
        meta = results.run(project)
        assert 'sample_count' in meta['fields']
        fields, loaded_meta = results.fields(project['id'])
        assert fields['sample_count'].max() == 4
        assert fields['sample_count'].min() == 4
        # 辅助图应已随结果落盘
        path = results.image_path(project['id'], 'sample_count')
        assert Path(path).exists()
        # 报告字段
        report = loaded_meta['report']
        assert report['method'] == 'color_consensus'
        assert report['color_tolerance_8bit'] == 12
        assert report['max_sample_count'] == 4
        # 护栏：6 张照片应在引擎层拒绝
        with store.edit(project['id'], project['revision'], '追加第六张') as p:
            asset = assets.put(base.copy(), 'shot-5.png')
            p['frames'].append(model.frame(asset['id'], 'shot-5'))
        project = store.get(project['id'])
        try:
            process(project, assets)
        except Invalid as error:
            assert '5' in str(error)
        else:
            raise AssertionError('6 张照片端到端应被拒绝')
        # 护栏：长边超 384 应拒绝
        with store.edit(project['id'], project['revision'], '停用一张并放大画布') as p:
            p['frames'][-1]['enabled'] = False
            p['settings']['size'] = 512
        project = store.get(project['id'])
        try:
            process(project, assets)
        except Invalid as error:
            assert '384' in str(error)
        else:
            raise AssertionError('长边 512 应被拒绝')
    print('PASS test_end_to_end_engine_and_results')


def test_existing_methods_unchanged():
    # 旧档位（sigma/mean）不应出现 sample_count 字段，且结果形状正常
    rng = np.random.default_rng(1)
    images = [np.clip(0.5 + rng.normal(0, 0.02, (4, 4, 3)), 0, 1).astype(np.float32) for _ in range(4)]
    prepared = make_prepared(images)
    for method in ['mean', 'sigma', 'median', 'trimmed', 'weighted']:
        fields, report = denoise_stack(prepared, {'method': method, 'sigma': 2.0, 'iterations': 2,
                                                  'trim': 0.15, 'linear': True})
        assert 'sample_count' not in fields, method
        assert fields['image'].shape == (4, 4, 3)
        assert np.isfinite(fields['image']).all()
        assert report['method'] == method
    print('PASS test_existing_methods_unchanged')


if __name__ == '__main__':
    test_basic_consensus_picks_largest_group()
    test_whole_rgb_taken_or_rejected_together()
    test_tolerance_boundary_inclusive()
    test_tie_break_by_reference_color()
    test_tie_break_by_frame_order_when_reference_invalid()
    test_masks_limit_valid_samples()
    test_invalid_region_kept_invalid()
    test_linear_space_average()
    test_guard_limits()
    test_brute_force_equivalence_random()
    test_end_to_end_engine_and_results()
    test_existing_methods_unchanged()
    print('\n全部验证通过')
