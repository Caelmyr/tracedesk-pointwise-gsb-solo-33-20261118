#!/usr/bin/env python3
"""自动转单线的 Flask 端到端验证。运行：python3 -I scripts/test_vectorize_http.py"""
import io
import os
import sys
import tempfile

_ENV_SITE = '/opt/pointwise-env/site-packages'
if os.path.isdir(_ENV_SITE) and _ENV_SITE not in sys.path:
    sys.path.insert(0, _ENV_SITE)

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from image_lab import create_app
from tracedesk.server import install


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print('ok -', msg)


tmp = tempfile.mkdtemp(prefix='td-e2e-')
app = create_app(os.path.join(tmp, 'prep'))
install(app, os.path.join(tmp, 'draw'))
client = app.test_client()

# 合成底图：粗线 + 闭环 + 分叉，最长边<=256
im = Image.new('RGBA', (200, 150), 'white')
d = ImageDraw.Draw(im)
d.line([(20, 120), (70, 60), (120, 110)], fill=(25, 25, 25, 255), width=7)
d.line([(70, 60), (75, 20)], fill=(25, 25, 25, 255), width=5)
d.ellipse((125, 40, 180, 100), outline=(25, 25, 25, 255), width=6)
buf = io.BytesIO()
im.save(buf, format='PNG')
r = client.post('/tracedesk/upload', data={'file': (io.BytesIO(buf.getvalue()), 'art.png')},
                content_type='multipart/form-data')
check(r.status_code == 200, '上传底图成功')
image_id = r.get_json()['image']['id']

r = client.post('/tracedesk/drawings', json={'name': '自动测试稿'})
check(r.status_code == 200, '新建图稿成功')
p = r.get_json()['drawing']
key, rev = p['id'], p['revision']

r = client.post(f'/tracedesk/drawings/{key}/edit',
                json={'action': 'image', 'image_id': image_id, 'revision': rev})
check(r.status_code == 200, '设置底图成功')
rev = r.get_json()['drawing']['revision']

# 先加一条人工路径
r = client.post(f'/tracedesk/drawings/{key}/edit',
                json={'action': 'add', 'name': '人工路径', 'points': [[10, 10], [50, 50]],
                      'closed': False, 'revision': rev})
check(r.status_code == 200, '人工描线照常保存')
rev = r.get_json()['drawing']['revision']
manual_paths = r.get_json()['drawing']['paths']
check(len(manual_paths) == 1, '采纳前存在1条人工路径')

# 阈值黑白预览
r = client.post(f'/tracedesk/drawings/{key}/auto-mask', json={'revision': rev, 'threshold': 128})
check(r.status_code == 200 and r.mimetype == 'image/png', '阈值预览返回PNG')
mask_im = Image.open(io.BytesIO(r.data))
check(mask_im.size == (200, 150), '阈值预览与画布同尺寸')

# 识别预览（不落库）
r = client.post(f'/tracedesk/drawings/{key}/auto-preview',
                json={'revision': rev, 'threshold': 128, 'mode': 'faithful'})
check(r.status_code == 200, '忠实预览生成')
prev = r.get_json()['preview']
check(prev['path_count'] >= 3, f'忠实预览含多条路径（{prev["path_count"]}）')
check(prev['closed_count'] >= 1, '忠实预览含闭环')
check(prev['revision'] == rev, '预览回传修订号')

# 预览不改变已保存的图稿
r = client.get(f'/tracedesk/drawings/{key}')
saved = r.get_json()['drawing']
check(len(saved['paths']) == 1 and saved['revision'] == rev, '预览阶段不落库不改修订号')

# 简洁预览
r = client.post(f'/tracedesk/drawings/{key}/auto-preview',
                json={'revision': rev, 'threshold': 128, 'mode': 'simple'})
check(r.status_code == 200, '简洁预览生成')
simple = r.get_json()['preview']
check(simple['node_count'] <= prev['node_count'], '简洁节点数不超过忠实')
check(simple['closed_count'] == prev['closed_count'], '简洁保留全部闭环')

# 旧修订号的预览必须失效
r = client.post(f'/tracedesk/drawings/{key}/auto-preview',
                json={'revision': rev - 1, 'threshold': 128, 'mode': 'faithful'})
check(r.status_code == 409, '过期修订号的预览被拒绝（旧预览不能盖掉新编辑）')

# 采纳：替换当前描线
r = client.post(f'/tracedesk/drawings/{key}/auto-apply',
                json={'revision': rev, 'threshold': 128, 'mode': 'faithful'})
check(r.status_code == 200, '采纳成功')
after = r.get_json()['drawing']
check(after['revision'] == rev + 1, '采纳产生一个新修订')
check(len(after['paths']) == prev['path_count'], '采纳后路径与预览一致')
check(all(set(q) == {'id', 'name', 'points', 'closed'} for q in after['paths']),
      '生成路径结构与手工路径同构（可被现有编辑器处理）')
check(not any(q['name'] == '人工路径' for q in after['paths']), '采纳替换掉原有全部路径')

# 一次撤销恢复原来的人工路径
r = client.post(f'/tracedesk/drawings/{key}/history',
                json={'revision': after['revision'], 'action': 'undo'})
check(r.status_code == 200, '撤销成功')
undone = r.get_json()['drawing']
check(len(undone['paths']) == 1 and undone['paths'][0]['name'] == '人工路径',
      '一次撤销恢复采纳前的全部路径')

# 采纳结果可导出 PNG / SVG / ZIP
r2 = client.post(f'/tracedesk/drawings/{key}/history',
                 json={'revision': undone['revision'], 'action': 'redo'})
check(r2.status_code == 200, '重做恢复采纳结果')
for suffix, mime in (('.png', 'image/png'), ('.svg', 'image/svg+xml'), ('.zip', 'application/zip')):
    rr = client.get(f'/tracedesk/drawings/{key}{suffix}')
    check(rr.status_code == 200 and rr.mimetype == mime, f'导出{suffix}正常')
svg_text = client.get(f'/tracedesk/drawings/{key}.svg').data.decode()
check(svg_text.count('<path') == prev['path_count'], 'SVG包含全部生成路径')
check('Z' in svg_text, 'SVG中闭环使用Z闭合')

# 采纳后的路径仍可走原有编辑动作（节点编辑/删除/平移）
cur = client.get(f'/tracedesk/drawings/{key}').get_json()['drawing']
qid = cur['paths'][0]['id']
r = client.post(f'/tracedesk/drawings/{key}/edit',
                json={'action': 'point', 'path_id': qid, 'index': 0, 'x': 12.5, 'y': 13.5,
                      'revision': cur['revision']})
check(r.status_code == 200, '生成路径支持节点编辑')
cur = r.get_json()['drawing']
r = client.post(f'/tracedesk/drawings/{key}/edit',
                json={'action': 'move', 'path_id': qid, 'dx': 1, 'dy': 0,
                      'revision': cur['revision']})
check(r.status_code == 200, '生成路径支持平移')
r = client.post(f'/tracedesk/drawings/{key}/edit',
                json={'action': 'delete', 'path_id': qid, 'revision': cur['revision'] + 1})
check(r.status_code == 200, '生成路径支持删除')

# 超256底图被拒绝
big = Image.new('RGBA', (257, 100), 'white')
bb = io.BytesIO()
big.save(bb, format='PNG')
r = client.post('/tracedesk/upload', data={'file': (io.BytesIO(bb.getvalue()), 'big.png')},
                content_type='multipart/form-data')
# 通用上传允许2048内；拒绝应发生在自动转单线
check(r.status_code == 200, '257px图仍可作为普通底图上传')
big_id = r.get_json()['image']['id']
r = client.post('/tracedesk/drawings', json={'name': '大稿', 'width': 257, 'height': 100})
big_key = r.get_json()['drawing']['id']
big_rev = r.get_json()['drawing']['revision']
r = client.post(f'/tracedesk/drawings/{big_key}/edit',
                json={'action': 'image', 'image_id': big_id, 'revision': big_rev})
big_rev = r.get_json()['drawing']['revision']
r = client.post(f'/tracedesk/drawings/{big_key}/auto-preview',
                json={'revision': big_rev, 'threshold': 128, 'mode': 'faithful'})
check(r.status_code == 400 and '256' in r.get_json()['error'], '超256px底图拒绝自动转单线')
r = client.post(f'/tracedesk/drawings/{big_key}/auto-apply',
                json={'revision': big_rev, 'threshold': 128, 'mode': 'faithful',
                      'image_id': big_id})
check(r.status_code == 400, '采纳接口同样拒绝超尺寸底图')

# 无底图时拒绝
r = client.post('/tracedesk/drawings', json={'name': '空稿'})
empty = r.get_json()['drawing']
r = client.post(f'/tracedesk/drawings/{empty["id"]}/auto-preview',
                json={'revision': empty['revision'], 'threshold': 128, 'mode': 'faithful'})
check(r.status_code == 400 and '底图' in r.get_json()['error'], '无底图时拒绝自动转单线')

# 非法参数
r = client.post(f'/tracedesk/drawings/{key}/auto-preview',
                json={'revision': cur['revision'] + 1, 'threshold': 300, 'mode': 'weird'})
check(r.status_code in (400, 409), '非法阈值/模式被拒绝')

print('\n端到端全部通过。')
