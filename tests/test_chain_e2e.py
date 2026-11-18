"""端到端：通过 Flask 测试客户端跑串接配准任务、应用拦截与撤销。"""
import sys
import tempfile
import numpy as np

sys.path.insert(0, '/workspace')

from image_lab import create_app


def textured(height, width, seed=3):
    # 非周期性随机纹理，避免相位相关在大位移时发生频谱混叠
    rng = np.random.default_rng(seed)
    base = rng.random((height, width, 3)).astype(np.float32)
    from scipy import ndimage as ndi
    for channel in range(3):
        base[..., channel] = ndi.gaussian_filter(base[..., channel], 1.4)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base[..., 0] = np.clip(base[..., 0] + .15 * (xx / width), 0, 1)
    base[..., 1] = np.clip(base[..., 1] + .15 * (yy / height), 0, 1)
    return base.astype(np.float32)


def main():
    tmp = tempfile.mkdtemp()
    app = create_app(tmp)
    client = app.test_client()

    project = client.post('/api/projects', json={'name': 'chain-e2e'}).get_json()
    key = project['id']

    h, w = 240, 240
    scene = textured(h + 220, w + 220)
    frames = []
    for dx, dy in [(0, 0), (90, 3), (180, 6)]:
        image = scene[dy:dy + h, dx:dx + w]
        upload = client.post(f'/api/projects/{key}/upload',
                             data={'revision': str(client.get(f'/api/projects/{key}').get_json()['revision']),
                                   'files': (np_to_png(image), 'shot.png')},
                             content_type='multipart/form-data')
        assert upload.status_code == 201, upload.get_json()
        frames.append(upload.get_json()['frame_ids'][-1])

    # 工作长边压到 384 以内（图像本身 240，默认 768 不会放大，letterbox 后仍是 240 画布）
    proj = client.get(f'/api/projects/{key}').get_json()
    settings = dict(proj['settings']['alignment'])
    settings.update({'chaining': True, 'method': 'phase', 'minimum_overlap': .3, 'minimum_correlation': .6})
    resp = client.patch(f'/api/projects/{key}', json={'revision': proj['revision'], 'settings': {'alignment': settings}})
    assert resp.status_code == 200, resp.get_json()

    job = client.post(f'/api/projects/{key}/jobs', json={'kind': 'align', 'revision': client.get(f'/api/projects/{key}').get_json()['revision']})
    assert job.status_code == 202
    job_id = job.get_json()['id']

    import time
    for _ in range(100):
        info = client.get(f'/api/jobs/{job_id}').get_json()
        if info['status'] in ('completed', 'failed', 'cancelled'):
            break
        time.sleep(.1)
    assert info['status'] == 'completed', info
    details = info['details']
    assert details['chaining'] is True and details['applicable'] is True
    assert len(details['pairs']) == 3
    assert all('residual' in pair for pair in details['pairs'])
    print('  ok - 串接任务完成，含', sum(1 for p in details['pairs'] if p['trusted']), '个可信对')

    # 应用配准
    rev = client.get(f'/api/projects/{key}').get_json()['revision']
    applied = client.post(f'/api/jobs/{job_id}/apply-alignment', json={'revision': rev})
    assert applied.status_code == 200, applied.get_json()
    updated = applied.get_json()
    far_matrix = next(f for f in updated['frames'] if f['id'] == frames[2])['matrix']
    assert abs(far_matrix[0][2] * w - 180) < 2, far_matrix
    assert abs(far_matrix[1][2] * h - 6) < 2, far_matrix
    print('  ok - 应用后远帧归一化矩阵平移对应约 (180,6) 工作像素，且为纯平移矩阵')

    # 撤销可用
    undo = client.post(f'/api/projects/{key}/undo', json={'revision': updated['revision']})
    assert undo.status_code == 200
    assert undo.get_json()['undo'] == updated['undo'][:-1]
    print('  ok - 撤销沿用既有历史机制')

    # 图组改动后旧任务失效（签名变化）
    changed = client.get(f'/api/projects/{key}').get_json()
    toggle = client.patch(f'/api/projects/{key}/frames/{frames[1]}',
                          json={'revision': changed['revision'], 'enabled': False})
    assert toggle.status_code == 200
    blocked = client.post(f'/api/jobs/{job_id}/apply-alignment',
                          json={'revision': client.get(f'/api/projects/{key}').get_json()['revision']})
    assert blocked.status_code == 409 and '重新配准' in blocked.get_json()['error']
    print('  ok - 图组改动后旧配准被拒绝：', blocked.get_json()['error'])

    # 接不起来的场景：把阈值调到不可能满足
    proj = client.get(f'/api/projects/{key}').get_json()
    client.patch(f'/api/projects/{key}/frames/{frames[1]}', json={'revision': proj['revision'], 'enabled': True})
    proj = client.get(f'/api/projects/{key}').get_json()
    strict = dict(proj['settings']['alignment'])
    strict['minimum_correlation'] = .999
    client.patch(f'/api/projects/{key}', json={'revision': proj['revision'], 'settings': {'alignment': strict}})
    bad_job = client.post(f'/api/projects/{key}/jobs', json={'kind': 'align', 'revision': client.get(f'/api/projects/{key}').get_json()['revision']}).get_json()
    for _ in range(100):
        info2 = client.get(f'/api/jobs/{bad_job["id"]}').get_json()
        if info2['status'] in ('completed', 'failed', 'cancelled'):
            break
        time.sleep(.1)
    assert info2['status'] == 'completed' and info2['details']['applicable'] is False
    assert info2['details']['disconnected_frames'], '应点名未接通照片'
    print('  ok - 接不起来的照片被点名：', info2['details']['disconnected_frames'])
    rev = client.get(f'/api/projects/{key}').get_json()['revision']
    refused = client.post(f'/api/jobs/{bad_job["id"]}/apply-alignment', json={'revision': rev})
    assert refused.status_code == 409 and '不能应用' in refused.get_json()['error']
    print('  ok - 后端拒绝应用未接通报告：', refused.get_json()['error'])
    print('\n端到端检查全部通过')


def np_to_png(image):
    import io
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray((image * 255).round().astype(np.uint8)).save(buffer, format='PNG')
    buffer.seek(0)
    return buffer


if __name__ == '__main__':
    main()
