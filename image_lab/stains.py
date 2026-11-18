import base64
import cv2
import numpy as np
from . import model
from . import validation as v
from .codecs import png_bytes
from .stain_fill import analyze, fill_image, overlay_image, changed_outside, suggest, MAX_IMAGE_EDGE

SAMPLE_NAME = '旧织物 · 补片示范'
SAMPLE_NOTE = '合成示范图：程序生成的旧织物纹样，带一处焦痕。不是实拍照片。'
SAMPLE_STAIN = [322, 100, 382, 160]
SAMPLE_SEARCH = [200, 0, 494, 280]


def _data_url(image):
    return 'data:image/png;base64,' + base64.b64encode(png_bytes(image, 8)).decode('ascii')


def _canvas(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise v.Invalid('成图必须是H×W×3图像')
    if max(image.shape[:2]) > MAX_IMAGE_EDGE:
        raise v.Invalid(f'成图最长边为{max(image.shape[:2])}像素，超过本次修补上限{MAX_IMAGE_EDGE}像素；请把工作长边设为{MAX_IMAGE_EDGE}并重新合成')
    return image


def _weave(size, cell, angle, phase):
    yy, xx = np.indices(size, dtype=np.float32)
    along = xx * np.cos(angle) + yy * np.sin(angle)
    return .5 + .5 * np.cos((along + phase) * np.pi * 2 / cell)


def builtin_sample(width=512, height=384):
    """Synthetic aged textile: a woven repeat with one burnt stain."""
    yy, xx = np.indices((height, width), dtype=np.float32)
    rows, columns = int(np.ceil(height / 128)) + 1, int(np.ceil(width / 128)) + 1
    motif = np.zeros((rows, columns, 3), np.float32)
    index = np.arange(rows * columns).reshape(rows, columns).astype(np.float32)
    motif[:, :, 0] = .5 + .5 * np.sin(index * 1.7)
    motif[:, :, 1] = .5 + .5 * np.cos(index * 2.3 + 1)
    motif[:, :, 2] = .5 + .5 * np.sin(index * .9 + 2)
    field = cv2.resize(motif, (width, height), interpolation=cv2.INTER_CUBIC)
    field = .55 * field + .45 * cv2.GaussianBlur(field, (0, 0), 6)
    thread = .55 * _weave((height, width), 16, .32, 0) + .45 * _weave((height, width), 21, 1.24, 6)
    linen, walnut = np.array([.62, .55, .43], np.float32), np.array([.31, .24, .18], np.float32)
    sample = linen * (0.72 + 0.22 * thread[:, :, None]) + walnut * field * .55
    sample *= (1 - (yy / height) * .12)[:, :, None]
    stain_x, stain_y, radius = 352, 130, 28
    distance = np.hypot(xx - stain_x, yy - (stain_y + (xx - stain_x) * .1))
    falloff = np.clip((radius - distance) / 6, 0, 1)[:, :, None]
    soot = np.array([.12, .08, .05], np.float32)
    sample = sample * (1 - falloff * .92) + soot * falloff * .92
    grain = np.sin(xx * 12.9898 + yy * 78.233) * 43758.5453
    sample += (grain - np.floor(grain) - .5)[:, :, None] * .012
    return np.clip(sample, 0, 1).astype(np.float32)


class Stains:
    def __init__(self, store, assets, results):
        self.store = store
        self.assets = assets
        self.results = results

    def _current(self, key, result_id=None):
        fields, meta = self.results.fields(key, result_id, not bool(result_id))
        return _canvas(fields['image']), fields['valid'], meta

    def create_sample(self):
        """Create a project whose composited result is the synthetic old textile."""
        existing = next((row for row in self.store.list() if row['name'] == SAMPLE_NAME), None)
        if existing and existing['current_result']:
            return {'project': self.store.get(existing['id']), 'stain': SAMPLE_STAIN, 'search': SAMPLE_SEARCH}
        project = self.store.create(SAMPLE_NAME)
        image = builtin_sample()
        asset = self.assets.put(image, SAMPLE_NAME + '.png', {'synthetic': True, 'exif': {}})
        with self.store.edit(project['id'], 0, '载入旧织物修补示范') as current:
            current['settings']['mode'] = 'reference'
            current['settings']['size'] = max(image.shape[:2])
            current['notes'] = SAMPLE_NOTE
            frame = model.frame(asset['id'], SAMPLE_NAME, 1.)
            current['frames'].append(frame)
            current['reference'] = frame['id']
        current = self.store.get(project['id'])
        self.results.run(current)
        return {'project': self.store.get(project['id']), 'stain': SAMPLE_STAIN, 'search': SAMPLE_SEARCH}

    def auto(self, key, body):
        data = v.object_value(body, '自动查找参数')
        image, _, _ = self._current(key, data.get('result_id'))
        return suggest(image)

    def _analysis(self, key, data):
        image, valid, meta = self._current(key, data.get('result_id'))
        stain = data.get('stain')
        if stain is None:
            raise v.Invalid('请先框出要补的污点')
        return image, valid, meta, analyze(image, valid, stain, data.get('search'), meta['id'])

    def preview(self, key, body):
        data = v.object_value(body, '修补参数')
        image, valid, meta, analysis = self._analysis(key, data)
        response = {'available': analysis['available'], 'analysis': analysis, 'result': self._meta(meta), 'reason': analysis.get('reason')}
        if not analysis['available'] and analysis.get('stain') and analysis.get('search'):
            response['suggestion'] = suggest(image)
        if not analysis['available']:
            return response
        stain_bounds = analysis['stain']['bounds']
        patch = analysis['patch']['bounds']
        filled = fill_image(image, stain_bounds, patch)
        response.update({
            'patch_png': _data_url(self._crop(image, analysis['patch']['outer_bounds'])),
            'overlay_png': _data_url(overlay_image(image, stain_bounds, analysis['search']['bounds'], patch)),
            'filled_png': _data_url(filled),
            'focus_png': _data_url(self._crop(filled, self._focus_bounds(analysis))),
            'verification': {
                'changed_pixels': int((np.abs(filled - image).max(axis=2) > 1e-6).sum()),
                'stain_pixels': analysis['stain']['width'] * analysis['stain']['height'],
                'changed_outside_stain': changed_outside(image, filled, stain_bounds),
                'total_pixels': int(image.shape[0] * image.shape[1]),
                'original_result_unchanged': True,
            },
        })
        return response

    def export(self, key, body):
        data = v.object_value(body, '导出参数')
        image, valid, meta, analysis = self._analysis(key, data)
        if not analysis['available']:
            raise v.Invalid(analysis.get('reason', '没有可用补片'))
        if data.get('scheme_id') and data['scheme_id'] != analysis['scheme_id']:
            raise v.Invalid('预览时的方案已变化，请重新预览再导出')
        bits = v.choice(data.get('bits', 8), 'PNG位深', [8, 16])
        stain_bounds, patch = analysis['stain']['bounds'], analysis['patch']['bounds']
        filled = fill_image(image, stain_bounds, patch)
        if changed_outside(image, filled, stain_bounds):
            raise v.Invalid('检测到污点框外像素发生变化，已停止导出')
        report = self.report(key, data) | {'format': 'tracedesk-stain-fill-v1', 'only_stain_pixels_replaced': True, 'note': '导出采用与预览相同的方案；工程设置、原始素材和已存合成结果均未修改。'}
        return {'png': png_bytes(filled, bits), 'report': report}

    def report(self, key, body):
        data = v.object_value(body, '报告参数')
        _, _, meta, analysis = self._analysis(key, data)
        return analysis | {'source_result': self._meta(meta)}

    def _meta(self, meta):
        return {'result_id': meta['id'], 'width': meta['width'], 'height': meta['height'], 'created_at': meta['created_at'], 'stale': meta['stale']}

    def _crop(self, image, bounds):
        x0, y0, x1, y1 = [int(item) for item in bounds]
        x0, y0 = max(0, min(x0, image.shape[1] - 1)), max(0, min(y0, image.shape[0] - 1))
        x1, y1 = max(x0 + 1, min(x1, image.shape[1])), max(y0 + 1, min(y1, image.shape[0]))
        return image[y0:y1, x0:x1].copy()

    def _focus_bounds(self, analysis):
        stain, patch = analysis['stain']['bounds'], analysis['patch']['bounds']
        margin = max(8, analysis['stain']['width'], analysis['stain']['height']) // 2
        return [min(stain[0], patch[0]) - margin, min(stain[1], patch[1]) - margin, max(stain[2], patch[2]) + margin, max(stain[3], patch[3]) + margin]
