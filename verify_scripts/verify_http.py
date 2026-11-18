#!/usr/bin/env python3
"""HTTP 层集成测试：整组颜色一致降噪的设置保存、合成与取样张数诊断图。"""
import sys
import tempfile

sys.path.insert(0, '/workspace')

import numpy as np

from image_lab import create_app
from image_lab import model


def main():
    with tempfile.TemporaryDirectory(dir='/workspace/verify_data') as data_dir:
        app = create_app(data_dir)
        client = app.test_client()

        # 创建设置为 color_consensus 的工程
        created = client.post('/api/projects', json={'name': 'HTTP集成测试'}).get_json()
        key = created['id']

        # 上传 5 张同尺寸测试图（最后一张带反光怪色）
        base = (np.zeros((300, 400, 3), np.float32) + 0.45)
        files = []
        for index in range(5):
            img = base + np.float32(0.004 * (index - 2))
            if index == 4:
                img[:] = np.array([0.95, 0.3, 0.2], np.float32)
            files.append((f'f{index}.png', _png(img), 'image/png'))
        from werkzeug.datastructures import MultiDict
        form = MultiDict([('revision', '0')] +
                         [('files', (body, name, mime)) for name, body, mime in files])
        resp = client.post(f'/api/projects/{key}/upload', data=form,
                           content_type='multipart/form-data')
        assert resp.status_code in (200, 201), resp.get_json()
        project = resp.get_json()['project']

        # 保存 color_consensus 设置（含 8 位容差），并切换到多帧降噪模式
        resp = client.patch(f'/api/projects/{key}', json={
            'revision': project['revision'],
            'settings': {'mode': 'denoise', 'size': 384,
                         'denoise': {'method': 'color_consensus', 'color_tolerance': 12, 'linear': True}}})
        assert resp.status_code == 200, resp.get_json()
        project = resp.get_json()
        denoise = project['settings']['denoise']
        assert denoise['method'] == 'color_consensus' and denoise['color_tolerance'] == 12
        print('PASS 设置保存与校验')

        # 容差超范围应被拒绝
        resp = client.patch(f'/api/projects/{key}', json={
            'revision': project['revision'],
            'settings': {'denoise': {'method': 'color_consensus', 'color_tolerance': 300}}})
        assert resp.status_code == 400
        print('PASS 容差范围校验')

        # 直接同步触发合成走任务接口；这里改用结果运行链路的 HTTP 任务
        resp = client.post(f'/api/projects/{key}/jobs', json={'revision': project['revision'], 'kind': 'process'})
        assert resp.status_code in (200, 201, 202), resp.get_json()
        job_id = resp.get_json()['id']
        import time
        for _ in range(100):
            job = client.get(f'/api/jobs/{job_id}').get_json()
            if job['status'] in ('completed', 'failed', 'cancelled', 'interrupted'):
                break
            time.sleep(0.1)
        assert job['status'] == 'completed', job
        print('PASS 合成任务完成')

        # 结果字段含 sample_count
        meta = client.get(f'/api/projects/{key}/current').get_json()
        assert 'sample_count' in meta['fields']
        assert meta['report']['method'] == 'color_consensus'
        assert meta['report']['max_sample_count'] == 4

        # 诊断图：实际取样张数
        resp = client.post(f'/api/projects/{key}/diagnostic-image', json={'kind': 'sample_count'})
        assert resp.status_code == 200, resp.get_json()
        assert resp.mimetype == 'image/png' and len(resp.data) > 100
        print('PASS 取样张数诊断图')

        # 辅助图静态入口也可取
        resp = client.get(f'/api/projects/{key}/result-image/sample_count')
        assert resp.status_code == 200 and resp.mimetype == 'image/png'
        print('PASS 辅助图结果入口')

        # 切回普通降噪后，sample_count 诊断应明确报错而非静默
        project = client.get(f'/api/projects/{key}').get_json()
        resp = client.patch(f'/api/projects/{key}', json={
            'revision': project['revision'],
            'settings': {'mode': 'denoise', 'denoise': {'method': 'sigma'}}})
        assert resp.status_code == 200
        project = resp.get_json()
        job = client.post(f'/api/projects/{key}/jobs',
                          json={'revision': project['revision'], 'kind': 'process'}).get_json()
        for _ in range(100):
            done = client.get(f'/api/jobs/{job["id"]}').get_json()
            if done['status'] in ('completed', 'failed', 'cancelled', 'interrupted'):
                break
            time.sleep(0.1)
        assert done['status'] == 'completed', done
        resp = client.post(f'/api/projects/{key}/diagnostic-image', json={'kind': 'sample_count'})
        assert resp.status_code == 400
        print('PASS 旧档位无取样张数图时明确报错')

        # 6 张照片护栏（HTTP 任务失败并带中文错误）
        project = client.get(f'/api/projects/{key}').get_json()
        extra = _png(base)
        form = MultiDict([('revision', str(project['revision'])),
                          ('files', (extra, 'f5.png', 'image/png'))])
        resp = client.post(f'/api/projects/{key}/upload', data=form,
                           content_type='multipart/form-data')
        assert resp.status_code in (200, 201)
        project = resp.get_json()['project']
        resp = client.patch(f'/api/projects/{key}', json={
            'revision': project['revision'],
            'settings': {'mode': 'denoise', 'size': 384,
                         'denoise': {'method': 'color_consensus', 'color_tolerance': 12}}})
        project = resp.get_json()
        job = client.post(f'/api/projects/{key}/jobs',
                          json={'revision': project['revision'], 'kind': 'process'}).get_json()
        for _ in range(100):
            done = client.get(f'/api/jobs/{job["id"]}').get_json()
            if done['status'] in ('completed', 'failed', 'cancelled', 'interrupted'):
                break
            time.sleep(0.1)
        assert done['status'] == 'failed' and '5' in done['error'], done
        print('PASS 超过5张护栏（HTTP）')

        # capabilities 仍正常暴露所有流程与降噪参数组
        cap = client.get('/api/capabilities').get_json()
        denoise_workflow = next(w for w in cap['workflows'] if w['key'] == 'denoise')
        assert denoise_workflow['settings_section'] == 'denoise'
        assert cap['settings']['denoise']['method'] == 'sigma'  # 新工程默认档位不变
        print('PASS capabilities 与默认档位保持不变')

    print('\nHTTP 集成测试全部通过')


def _png(image):
    from io import BytesIO
    from image_lab.codecs import png_bytes
    return BytesIO(png_bytes(image))


if __name__ == '__main__':
    main()
