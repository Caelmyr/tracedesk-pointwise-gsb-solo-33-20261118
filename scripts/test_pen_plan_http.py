#!/usr/bin/env python3
"""走笔安排 HTTP 端到端冒烟测试（临时数据目录，不触碰工作区数据）。"""
import glob
import json
import os
import re
import sys
import tempfile

for extra in glob.glob('/opt/*/site-packages'):
    if extra not in sys.path:
        sys.path.append(extra)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_lab import create_app  # noqa: E402
from tracedesk.server import install  # noqa: E402

root = tempfile.mkdtemp(prefix='td-e2e-')
app = create_app(os.path.join(root, 'preparation'))
install(app, os.path.join(root, 'drawings'))
client = app.test_client()

r = client.post('/tracedesk/drawings', json={'name': '走笔测试', 'width': 256, 'height': 192})
assert r.status_code == 200, r.data
key = r.get_json()['drawing']['id']

paths = [
    {'name': '开线A', 'points': [[200, 150], [220, 160]], 'closed': False},
    {'name': '开线B', 'points': [[30, 40], [60, 20]], 'closed': False},
    {'name': '闭环C', 'points': [[120, 90], [150, 90], [150, 120], [120, 120]],
     'closed': True},
]
for i, q in enumerate(paths):
    r = client.post(f'/tracedesk/drawings/{key}/edit',
                    json={'action': 'add', 'revision': i, **q})
    assert r.status_code == 200, r.data
rev = r.get_json()['drawing']['revision']
assert rev == 3

# 1. 预览
r = client.post(f'/tracedesk/drawings/{key}/pen-plan', json={'revision': rev})
assert r.status_code == 200, r.data
plan = r.get_json()['plan']
assert plan['revision'] == rev
assert plan['path_count'] == 3
assert len(plan['steps']) == 3 and len(plan['pen_up']) == 3
assert plan['moved_closed_count'] <= 1
assert plan['planned_travel'] <= plan['original_travel'] + 1e-9
assert plan['pen_up'][0]['from'] == [0.0, 0.0]
print('预览OK：原空走', plan['original_travel'], '新空走', plan['planned_travel'],
      '顺序', [(s['order'], s['name']) for s in plan['steps']])

# 2. 下载走笔SVG
r = client.get(f'/tracedesk/drawings/{key}/pen-plan.svg?revision={rev}')
assert r.status_code == 200
assert r.headers['Content-Type'].startswith('image/svg+xml')
assert r.headers['Content-Disposition'].startswith('attachment')
svg = r.data.decode()
ids = [s['path_id'] for s in plan['steps']]
tags = re.findall(r'data-path-id="([^"]+)"', svg)
assert tags == ids, (tags, ids)
assert svg.count(' Z') == 1
assert 'data-plan-order="3"' in svg
print('走笔SVG下载OK，路径顺序与预览一致')

# 3. 图稿改动后旧修订号下载应 409
r = client.post(f'/tracedesk/drawings/{key}/edit',
                json={'action': 'settings', 'revision': rev, 'line_width': 3})
assert r.status_code == 200
new_rev = r.get_json()['drawing']['revision']
r = client.get(f'/tracedesk/drawings/{key}/pen-plan.svg?revision={rev}')
assert r.status_code == 409 and r.get_json()['type'] == 'revision_conflict'
r = client.post(f'/tracedesk/drawings/{key}/pen-plan', json={'revision': rev})
assert r.status_code == 409 and r.get_json()['type'] == 'revision_conflict'
# 新修订号恢复可用
r = client.post(f'/tracedesk/drawings/{key}/pen-plan', json={'revision': new_rev})
assert r.status_code == 200
print('修订号冲突OK：改动后旧预览被拒绝，重新预览可用')

# 4. 原有 SVG/PNG 导出不受影响，且仍是原路径顺序
r = client.get(f'/tracedesk/drawings/{key}.svg')
old_svg = r.data.decode()
original_ids = [q['id'] for q in client.get(f'/tracedesk/drawings/{key}')
                .get_json()['drawing']['paths']]
assert re.findall(r'data-path-id="([^"]+)"', old_svg) == original_ids
assert 'data-plan-order' not in old_svg
r = client.get(f'/tracedesk/drawings/{key}.png')
assert r.status_code == 200 and r.data[:8] == b'\x89PNG\r\n\x1a\n'
r = client.get(f'/tracedesk/drawings/{key}.zip')
assert r.status_code == 200 and r.data[:2] == b'PK'
print('原 SVG/PNG/ZIP 导出保持原样')

# 5. 规模限制：7 条路径拒绝
for j in range(4):
    client.post(f'/tracedesk/drawings/{key}/edit',
                json={'action': 'add', 'revision': new_rev + j,
                      'name': f'补{j}', 'points': [[1, 1], [2, 2]], 'closed': False})
r = client.post(f'/tracedesk/drawings/{key}/pen-plan', json={'revision': new_rev + 4})
assert r.status_code == 400 and '6' in r.get_json()['error']
print('超过6条路径拒绝：', r.get_json()['error'])

# 6. 闭环 11 节点拒绝（新稿，避免上面的7条限制）
r = client.post('/tracedesk/drawings', json={'name': '大闭环', 'width': 256, 'height': 192})
key2 = r.get_json()['drawing']['id']
client.post(f'/tracedesk/drawings/{key2}/edit',
            json={'action': 'add', 'revision': 0, 'name': '大环', 'closed': True,
                  'points': [[i + 5, i + 5] for i in range(11)]})
r = client.post(f'/tracedesk/drawings/{key2}/pen-plan', json={'revision': 1})
assert r.status_code == 400 and '10' in r.get_json()['error']
print('闭环超10节点拒绝：', r.get_json()['error'])

# 7. 空稿拒绝
r = client.post('/tracedesk/drawings', json={'name': '空稿'})
key3 = r.get_json()['drawing']['id']
r = client.post(f'/tracedesk/drawings/{key3}/pen-plan', json={'revision': 0})
assert r.status_code == 400
print('空稿拒绝：', r.get_json()['error'])

print('端到端全部通过，临时目录：', root)
