import io
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _extra in ('/opt/pointwise-env/site-packages',):
    if os.path.isdir(_extra) and _extra not in sys.path:
        sys.path.append(_extra)

from image_lab.validation import Invalid
from tracedesk import morph
from tracedesk.morph import _contact, _cross, frame_valid, morph_paths, add_morph_path


def path(pid, name, points, closed=True):
    return {'id': pid, 'name': name, 'points': points, 'closed': closed}


def drawing(paths, width=100, height=100):
    return {'name': 't', 'width': width, 'height': height, 'image_id': None,
            'image_opacity': .35, 'stroke': '#256b72', 'line_width': 2,
            'paths': paths, 'undo': [], 'redo': []}


SQUARE = [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]]


class GeometryTest(unittest.TestCase):
    def test_contact_disjoint(self):
        self.assertIsNone(_contact((0, 0), (1, 0), (2, 0), (3, 0), 1e-9, 1e-6))

    def test_contact_proper_cross(self):
        hit = _contact((0, 0), (2, 2), (0, 2), (2, 0), 1e-9, 1e-6)
        self.assertEqual(hit[0], 'point')
        self.assertAlmostEqual(hit[1][0], 1.0)
        self.assertAlmostEqual(hit[1][1], 1.0)

    def test_contact_endpoint_touch(self):
        hit = _contact((0, 0), (1, 0), (1, 0), (1, 1), 1e-9, 1e-6)
        self.assertEqual(hit, ('point', (1, 0)))

    def test_contact_collinear_overlap(self):
        self.assertEqual(_contact((0, 0), (2, 0), (1, 0), (3, 0), 1e-9, 1e-6)[0], 'overlap')
        # 共线但只在端点相接
        self.assertEqual(_contact((0, 0), (1, 0), (1, 0), (2, 0), 1e-9, 1e-6), ('point', (1, 0)))

    def test_contact_endpoint_inside_other_segment(self):
        self.assertEqual(_contact((0, 0), (4, 0), (2, 0), (2, 2), 1e-9, 1e-6)[0], 'overlap')

    def test_frame_valid_simple_and_crossing(self):
        self.assertTrue(frame_valid(SQUARE, 1e-9, 1e-6))
        bowtie = [[0.0, 0.0], [4.0, 4.0], [4.0, 0.0], [0.0, 4.0]]
        self.assertFalse(frame_valid(bowtie, 1e-9, 1e-6))

    def test_frame_valid_duplicate_point(self):
        self.assertFalse(frame_valid([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], 1e-9, 1e-6))

    def test_frame_valid_nonadjacent_touch(self):
        # 两个三角形顶点相碰的非简单闭环（节点 0 与节点 2 重合已被重复检查覆盖；
        # 这里构造边经过非相邻顶点）
        pts = [[0.0, 0.0], [4.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 4.0]]
        self.assertFalse(frame_valid(pts, 1e-9, 1e-6))

    def test_triangle_valid(self):
        tri = [[0.0, 0.0], [3.0, 0.0], [1.0, 2.0]]
        self.assertTrue(frame_valid(tri, 1e-9, 1e-6))

    def test_frame_valid_t_junction(self):
        # 顶点 3 落在非相邻边 0-1 内部（T 型相碰），且不与任何顶点重合
        pts = [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [2.0, 0.0], [4.0, 4.0]]
        self.assertFalse(frame_valid(pts, 1e-9, 1e-6))


class MorphTest(unittest.TestCase):
    def test_identity(self):
        d = drawing([path('a', 'A', SQUARE), path('b', 'B', SQUARE)])
        r = morph_paths(d, 'a', 'b')
        self.assertEqual(r['node_count'], 4)
        self.assertEqual(r['candidate_count'], 8)
        # 方形反向配对在50%处塌缩为交叉/重节点，只有3种同向平移全程保持简单闭环
        self.assertEqual(r['valid_candidate_count'], 3)
        self.assertEqual(r['match'], {'shift': 0, 'reversed': False})
        self.assertAlmostEqual(r['max_move'], 0.0, places=6)
        for f in r['frames']:
            self.assertEqual(len(f['points']), 4)
            self.assertTrue(frame_valid(f['points'], 1e-9, 1e-6))
        self.assertEqual([f['t'] for f in r['frames']], [0.25, 0.5, 0.75])

    def test_rotation_finds_shift(self):
        b = [SQUARE[1], SQUARE[2], SQUARE[3], SQUARE[0]]
        d = drawing([path('a', 'A', SQUARE), path('b', 'B', b)])
        r = morph_paths(d, 'a', 'b')
        # b 从 a 的下一个角起笔，对应关系等价于 target_i = b[(i-1) mod n]
        self.assertEqual(r['match'], {'shift': 3, 'reversed': False})
        self.assertAlmostEqual(r['max_move'], 0.0, places=6)

    def test_reversed_direction(self):
        b = [SQUARE[0], SQUARE[3], SQUARE[2], SQUARE[1]]
        d = drawing([path('a', 'A', SQUARE), path('b', 'B', b)])
        r = morph_paths(d, 'a', 'b')
        self.assertTrue(r['match']['reversed'])
        self.assertEqual(r['match']['shift'], 0)
        self.assertAlmostEqual(r['max_move'], 0.0, places=6)

    def test_validation_rules(self):
        open_path = path('o', 'O', SQUARE, closed=False)
        with self.assertRaises(Invalid):
            morph_paths(drawing([open_path, path('b', 'B', SQUARE)]), 'o', 'b')
        small = path('s', 'S', [[0.0, 0.0], [1.0, 1.0]], closed=True)
        with self.assertRaises(Invalid):
            morph_paths(drawing([small, path('b', 'B', SQUARE)]), 's', 'b')
        tri = path('t', 'T', [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], closed=True)
        with self.assertRaises(Invalid):
            morph_paths(drawing([tri, path('b', 'B', SQUARE)]), 't', 'b')
        with self.assertRaises(Invalid):
            morph_paths(drawing([path('a', 'A', SQUARE)]), 'a', 'a')
        with self.assertRaises(Invalid):
            morph_paths(drawing([path('a', 'A', SQUARE)]), 'a', 'missing')

    def test_no_valid_pair_bowtie(self):
        # 自交蝴蝶结闭环：八种环保序配法的中间稿全部保持自交，无可用配法
        bowtie = [[0.0, 0.0], [4.0, 4.0], [4.0, 0.0], [0.0, 4.0]]
        shifted = [[8.0, 0.0], [12.0, 4.0], [12.0, 0.0], [8.0, 4.0]]
        d = drawing([path('a', 'A', bowtie), path('b', 'B', shifted)])
        with self.assertRaises(Invalid) as ctx:
            morph_paths(d, 'a', 'b')
        self.assertIn('找不到可用配法', str(ctx.exception))

    def test_optimization_order(self):
        a = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        b = [[0.0, 0.0], [12.0, 0.0], [12.0, 10.0], [0.0, 10.0]]
        d = drawing([path('a', 'A', a), path('b', 'B', b)])
        r = morph_paths(d, 'a', 'b')
        # 独立暴力重算，验证返回的配法在合格集合中字典序最优
        n = 4
        best = None
        for shift in range(n):
            for rev in (False, True):
                target = morph._ordered_target(b, shift, rev, n)
                dists = [math.hypot(u - x, v - y) for (x, y), (u, v) in zip(a, target)]
                frames = [morph._interpolate(a, target, t) for t in morph.STEPS]
                if all(frame_valid(f, 1e-9, 1e-6) for f in frames):
                    key = (max(dists), sum(dists), shift, rev)
                    if best is None or key < best:
                        best = key
        self.assertEqual((r['max_move'], r['total_move'], r['match']['shift'], r['match']['reversed']),
                         (round(best[0], 3), round(best[1], 3), best[2], best[3]))
        # 本案例同向平移0最优：最大移动 2px
        self.assertEqual(r['match'], {'shift': 0, 'reversed': False})
        self.assertAlmostEqual(r['max_move'], 2.0, places=6)

    def test_add_morph_path_uses_one_edit_step(self):
        d = drawing([path('a', '旧徽记', SQUARE),
                     path('b', '新徽记', [[12.0, 10.0], [32.0, 10.0], [32.0, 30.0], [12.0, 30.0]])])
        before = len(d['paths'])
        add_morph_path(d, {'old_id': 'a', 'new_id': 'b', 't': 0.5})
        self.assertEqual(len(d['paths']), before + 1)
        added = d['paths'][-1]
        self.assertTrue(added['closed'])
        self.assertIn('过渡50%', added['name'])
        self.assertEqual(added['points'][0], [11.0, 10.0])
        # edit() 内部只打一个检查点，一次撤销即可移除新增路径
        self.assertEqual(len(d['undo']), 1)
        from tracedesk.core import history
        history(d, 'undo')
        self.assertEqual(len(d['paths']), before)

    def test_add_morph_path_bad_t(self):
        d = drawing([path('a', 'A', SQUARE), path('b', 'B', SQUARE)])
        with self.assertRaises(Invalid):
            add_morph_path(d, {'old_id': 'a', 'new_id': 'b', 't': 0.3})
        with self.assertRaises(Invalid):
            add_morph_path(d, {'old_id': 'a', 'new_id': 'b', 't': 'x'})

    def test_twelve_node_limit(self):
        n = 12
        pts = [[50 + 40 * math.cos(2 * math.pi * i / n), 50 + 40 * math.sin(2 * math.pi * i / n)]
               for i in range(n)]
        shifted = pts[2:] + pts[:2]
        d = drawing([path('a', 'A', pts), path('b', 'B', shifted)])
        r = morph_paths(d, 'a', 'b')
        self.assertEqual(r['candidate_count'], 24)
        self.assertEqual(r['match']['shift'], n - 2)
        self.assertFalse(r['match']['reversed'])


class HttpTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        from image_lab import create_app
        from tracedesk.server import install
        app = create_app(os.path.join(self.tmp.name, 'prep'))
        install(app, os.path.join(self.tmp.name, 'draw'))
        self.client = app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def _new_drawing(self):
        r = self.client.post('/tracedesk/drawings', json={'name': 'm'})
        self.assertEqual(r.status_code, 200)
        return r.get_json()['drawing']

    def _add(self, key, rev, pid, name, points):
        r = self.client.post(f'/tracedesk/drawings/{key}/edit',
                             json={'revision': rev, 'action': 'add', 'path_id': pid,
                                   'name': name, 'points': points, 'closed': True})
        self.assertEqual(r.status_code, 200, r.data)
        return r.get_json()['drawing']

    def test_full_flow(self):
        d = self._new_drawing()
        key, rev = d['id'], d['revision']
        d = self._add(key, rev, 'a', '旧徽记', SQUARE)
        d = self._add(key, d['revision'], 'b', '新徽记',
                      [[12.0, 10.0], [32.0, 10.0], [32.0, 30.0], [12.0, 30.0]])
        a_id, b_id = d['paths'][0]['id'], d['paths'][1]['id']
        r = self.client.post(f'/tracedesk/drawings/{key}/morph/preview',
                             json={'old_id': a_id, 'new_id': b_id})
        self.assertEqual(r.status_code, 200)
        m = r.get_json()['morph']
        self.assertEqual(len(m['frames']), 3)
        # 加为普通路径
        r = self.client.post(f'/tracedesk/drawings/{key}/morph/add',
                             json={'revision': d['revision'], 'old_id': a_id,
                                   'new_id': b_id, 't': 0.5})
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertEqual(len(j['drawing']['paths']), 3)
        # 一次撤销移除新增
        r = self.client.post(f'/tracedesk/drawings/{key}/history',
                             json={'revision': j['drawing']['revision'], 'action': 'undo'})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.get_json()['drawing']['paths']), 2)

    def test_preview_failure_400_and_no_mutation(self):
        d = self._new_drawing()
        key = d['id']
        bowtie = [[0.0, 0.0], [4.0, 4.0], [4.0, 0.0], [0.0, 4.0]]
        shifted = [[8.0, 0.0], [12.0, 4.0], [12.0, 0.0], [8.0, 4.0]]
        d = self._add(key, d['revision'], 'a', 'A', bowtie)
        d = self._add(key, d['revision'], 'b', 'B', shifted)
        a_id, b_id = d['paths'][0]['id'], d['paths'][1]['id']
        r = self.client.post(f'/tracedesk/drawings/{key}/morph/preview',
                             json={'old_id': a_id, 'new_id': b_id})
        self.assertEqual(r.status_code, 400)
        self.assertIn('找不到可用配法', r.get_json()['error'])
        # 失败不改变图稿
        r = self.client.get(f'/tracedesk/drawings/{key}')
        self.assertEqual(len(r.get_json()['drawing']['paths']), 2)

    def test_stale_revision_rejected(self):
        d = self._new_drawing()
        key, rev = d['id'], d['revision']
        d = self._add(key, rev, 'a', '旧徽记', SQUARE)
        d = self._add(key, d['revision'], 'b', '新徽记',
                      [[12.0, 10.0], [32.0, 10.0], [32.0, 30.0], [12.0, 30.0]])
        a_id, b_id = d['paths'][0]['id'], d['paths'][1]['id']
        r = self.client.post(f'/tracedesk/drawings/{key}/morph/add',
                             json={'revision': rev, 'old_id': a_id, 'new_id': b_id, 't': 0.5})
        self.assertEqual(r.status_code, 409)


if __name__ == '__main__':
    unittest.main(verbosity=2)
