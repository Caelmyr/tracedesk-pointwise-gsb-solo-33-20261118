from copy import deepcopy
from itertools import combinations
import numpy as np
from . import model
from . import validation as v
from .arrays import luminance
from .preparation import prepare
from .regions import pixel_bounds
from .statistics import focus_statistics

REGION_SELECTION_LIMIT = 8


def _region_arguments(project, body):
    if project['settings']['mode'] not in ('fusion', 'hdr', 'reference'):
        raise v.Invalid('区域曝光帧筛选只适用于曝光融合/HDR/参考帧模式')
    available = model.enabled_frames(project)
    count = v.integer(body.get('count', min(3, len(available))), '保留照片数', 1, min(REGION_SELECTION_LIMIT, len(available)))
    low = v.number(body.get('low', .05), '有效曝光低点', .001, .4)
    high = v.number(body.get('high', .95), '有效曝光高点', .6, .999)
    if low >= high:
        raise v.Invalid('有效曝光低点必须低于高点')
    region_ids = [v.identifier(item) for item in v.list_value(body.get('region_ids', []), '检测区域', 2)]
    if not 1 <= len(region_ids) <= 2:
        raise v.Invalid('请选择一到两个检测区域')
    if len(set(region_ids)) != len(region_ids):
        raise v.Invalid('检测区域不能重复')
    defined = {region['id']: region for region in project.get('regions', [])}
    regions = []
    for region_id in region_ids:
        region = defined.get(region_id)
        if region is None:
            raise v.Invalid('检测区域不存在')
        regions.append(region)
    targets = v.list_value(body.get('targets', []), '区域目标覆盖率', 2)
    if len(targets) != len(regions):
        raise v.Invalid('每个检测区域都需要填写目标覆盖率')
    targets = [v.number(value, '区域目标覆盖率', 0, 1) for value in targets]
    return available, count, low, high, regions, targets


def _region_materials(project, assets, low, high, regions):
    """对齐后、裁切前的参考画布上逐帧有效曝光图与区域切片。"""
    copy = deepcopy(project)
    copy['settings']['size'] = min(384, copy['settings']['size'])
    prepared = prepare(copy, assets)
    quality_maps, rows = [], []
    for image, valid, frame in zip(prepared['images'], prepared['validities'], prepared['frames']):
        gray = luminance(image)
        support = valid > .99
        well = support & (gray >= low) & (gray <= high)
        quality_maps.append(well)
        middle = np.maximum(0, 1 - np.abs(gray - .5) * 2) * support
        rows.append({'frame_id': frame['id'], 'name': frame['name'], 'well_exposed_fraction': float(well.mean()), 'balanced_score': float(middle.mean()), 'sharpness': focus_statistics(image)['laplacian_variance']})
    shape = quality_maps[0].shape
    bounds_list = [pixel_bounds(region, shape) for region in regions]
    slices = [(slice(top, bottom), slice(left, right)) for left, top, right, bottom in bounds_list]
    region_pixels = [(bottom - top) * (right - left) for left, top, right, bottom in bounds_list]
    return prepared, quality_maps, rows, slices, region_pixels, bounds_list


def region_coverage_plan(project, assets, body):
    """在固定帧数与参考帧下，寻找让选中区域各自达标且整图覆盖率最高的组合。"""
    available, count, low, high, regions, targets = _region_arguments(project, body)
    prepared, quality_maps, rows, slices, region_pixels, bounds_list = _region_materials(project, assets, low, high, regions)
    frame_ids = [frame['id'] for frame in prepared['frames']]
    reference_index = prepared['reference_index']
    reference_id = prepared['frames'][reference_index]['id']
    full_well = np.stack(quality_maps)
    region_well = [full_well[(slice(None),) + part] for part in slices]

    def coverage(indices, source, pixels):
        return float(np.count_nonzero(np.any(source[list(indices)], axis=0))) / pixels

    all_indices = frozenset(range(len(frame_ids)))
    full_all = coverage(all_indices, full_well, full_well.shape[1] * full_well.shape[2])
    region_all = [coverage(all_indices, source, pixels) for source, pixels in zip(region_well, region_pixels)]
    others = [index for index in range(len(frame_ids)) if index != reference_index]
    best = None
    for picked in combinations(others, count - 1):
        indices = (reference_index,) + picked
        region_fractions = [coverage(indices, source, pixels) for source, pixels in zip(region_well, region_pixels)]
        if any(fraction < target for fraction, target in zip(region_fractions, targets)):
            continue
        full_fraction = coverage(indices, full_well, full_well.shape[1] * full_well.shape[2])
        balance = sum(rows[index]['balanced_score'] for index in indices)
        key = (full_fraction, balance, tuple(sorted(frame_ids[index] for index in indices)))
        if best is None or key > best[0]:
            best = (key, indices, region_fractions, full_fraction)
    base = {'revision': project['revision'], 'reference_frame_id': reference_id, 'count': count, 'low': low, 'high': high, 'enabled_pool_size': len(frame_ids), 'pool_limit': REGION_SELECTION_LIMIT,
            'regions': [{'region_id': region['id'], 'name': region['name'], 'target_coverage': target, 'pixel_bounds': bounds, 'all_frames_coverage': ceiling} for region, target, bounds, ceiling in zip(regions, targets, bounds_list, region_all)],
            'frames': rows}
    if best is None:
        base.update({'feasible': False, 'selected_frame_ids': [], 'coverage_fraction': 0., 'all_frame_coverage_fraction': full_all, 'note': '固定帧数下没有组合能让每个选中区域都达到目标覆盖率；未修改工程。'})
        return base
    _, indices, region_fractions, full_fraction = best
    selected_ids = [frame_ids[index] for index in sorted(indices)]
    for region, fraction in zip(base['regions'], region_fractions):
        region['coverage_fraction'] = fraction
    base.update({'feasible': True, 'selected_frame_ids': selected_ids, 'coverage_fraction': full_fraction, 'all_frame_coverage_fraction': full_all, 'note': '穷举全部含参考帧的固定张数组合：先保证逐区达标，再取整图覆盖率最高；覆盖按对齐后未裁切参考画布计算。'})
    return base


def apply_region_selection(store, assets, key, revision, body):
    current = store.get(key)
    if current['revision'] != revision:
        raise v.Conflict('工程已更新，请重新生成区域筛选建议')
    plan = region_coverage_plan(current, assets, body)
    if not plan['feasible']:
        raise v.Invalid('无可用方案：固定帧数无法让选中区域全部达标，工程未修改')
    selected = set(plan['selected_frame_ids'])
    with store.edit(key, revision, '按区域覆盖目标筛选照片') as project:
        for frame in project['frames']:
            frame['enabled'] = frame['id'] in selected
    return {'project': store.get(key), 'selection': plan}


def select_bracket(project, assets, body):
    available = model.enabled_frames(project)
    count = v.integer(body.get('count', min(3, len(available))), '保留照片数', 1, min(12, len(available)))
    low = v.number(body.get('low', .05), '有效曝光低点', .001, .4)
    high = v.number(body.get('high', .95), '有效曝光高点', .6, .999)
    copy = deepcopy(project)
    copy['settings']['size'] = min(384, copy['settings']['size'])
    prepared = prepare(copy, assets)
    images = prepared['images']
    validities = prepared['validities']
    covered = np.zeros(images[0].shape[:2], bool)
    quality_maps = []
    balanced_scores = []
    sharpness_scores = []
    rows = []
    for image, valid, frame in zip(images, validities, prepared['frames']):
        gray = luminance(image)
        support = valid > .99
        well = support & (gray >= low) & (gray <= high)
        quality_maps.append(well)
        middle = np.maximum(0, 1 - np.abs(gray - .5) * 2) * support
        balanced_scores.append(float(middle.mean()))
        sharpness = focus_statistics(image)['laplacian_variance']
        sharpness_scores.append(sharpness)
        rows.append({'frame_id': frame['id'], 'name': frame['name'], 'well_exposed_fraction': float(well.mean()), 'balanced_score': float(middle.mean()), 'sharpness': sharpness})
    first = int(np.argmax(balanced_scores))
    chosen = [first]
    covered |= quality_maps[first]
    incremental = [float(covered.mean())]
    while len(chosen) < count:
        candidates = []
        for index, quality in enumerate(quality_maps):
            if index in chosen:
                continue
            gain = float(np.mean(quality & ~covered))
            redundancy = float(np.mean(quality & covered))
            score = gain + balanced_scores[index] * .0001 + np.log1p(sharpness_scores[index]) * .00001 - redundancy * .000001
            candidates.append((score, index, gain))
        _, best, gain = max(candidates)
        chosen.append(best)
        covered |= quality_maps[best]
        incremental.append(gain)
    selected_ids = [prepared['frames'][index]['id'] for index in sorted(chosen)]
    reference = prepared['frames'][first]['id']
    union = np.any(np.stack(quality_maps), axis=0)
    selected_coverage = float(covered.mean())
    total_coverage = float(union.mean())
    loss = max(0, total_coverage - selected_coverage)
    return {'selected_frame_ids': selected_ids, 'suggested_reference': reference, 'coverage_fraction': selected_coverage, 'all_frame_coverage_fraction': total_coverage, 'coverage_loss_fraction': loss, 'incremental_gains': incremental, 'frames': rows, 'note': '贪心保留互补的未截断像素；不适用于景深/降噪序列筛选，也不保证HDR辐射精度最优。'}


def apply_selection(store, assets, key, revision, body):
    project = store.get(key)
    if project['settings']['mode'] not in ('fusion', 'hdr', 'reference'):
        raise v.Invalid('自动曝光帧筛选只适用于曝光融合/HDR/参考帧模式')
    plan = select_bracket(project, assets, body)
    selected = set(plan['selected_frame_ids'])
    with store.edit(key, revision, '按曝光覆盖筛选照片') as current:
        for frame in current['frames']:
            frame['enabled'] = frame['id'] in selected
        current['reference'] = plan['suggested_reference']
    return {'project': store.get(key), 'selection': plan}


def exposure_coverage_map(project, assets, body):
    copy = deepcopy(project)
    copy['settings']['size'] = min(768, copy['settings']['size'])
    prepared = prepare(copy, assets)
    low = v.number(body.get('low', .05), '覆盖低点', 0, .4)
    high = v.number(body.get('high', .95), '覆盖高点', .6, 1)
    counts = np.zeros(prepared['images'][0].shape[:2], np.float32)
    reliability = np.zeros_like(counts)
    for image, valid in zip(prepared['images'], prepared['validities']):
        gray = luminance(image)
        exposed = (gray >= low) & (gray <= high)
        counts += exposed * valid
        reliability = np.maximum(reliability, np.maximum(0, 1 - np.abs(gray - .5) * 2) * valid)
    missing = (counts < .5).astype(np.float32)
    return {'coverage': counts / len(prepared['images']), 'unrecoverable': missing, 'best_exposedness': reliability}
