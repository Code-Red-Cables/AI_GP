"""GateNet maps four inner corners onto the existing eight-keypoint layout.

The friend's model is observe-only by default. These tests pin the mapping and
the panel so GateNet and YOLO are never drawn on the same frame.
"""
from __future__ import annotations

import unittest

import numpy as np

from vision.gatenet_detector import (
    INNER_KP_OFFSET,
    inner_keypoints_from_corners,
)


class MappingTests(unittest.TestCase):
    def test_inner_ring_is_keypoints_4_to_7(self):
        corners = np.array([
            [100.0, 80.0],   # TL
            [260.0, 80.0],   # TR
            [260.0, 240.0],  # BR
            [100.0, 240.0],  # BL
        ])
        scores = np.array([0.9, 0.9, 0.9, 0.9])
        kps, conf = inner_keypoints_from_corners(
            corners, scores, threshold=0.80,
        )
        self.assertEqual(len(kps), 8)
        for i in range(4):
            self.assertEqual(kps[i], [0.0, 0.0])
            self.assertEqual(conf[i], 0.0)
        self.assertEqual(kps[4], [100.0, 80.0])
        self.assertEqual(kps[5], [260.0, 80.0])
        self.assertEqual(kps[6], [260.0, 240.0])
        self.assertEqual(kps[7], [100.0, 240.0])
        self.assertEqual(INNER_KP_OFFSET, 4)

    def test_weak_corners_are_unseen_not_at_the_origin_as_a_detection(self):
        corners = np.array([
            [100.0, 80.0],
            [260.0, 80.0],
            [260.0, 240.0],
            [100.0, 240.0],
        ])
        scores = np.array([0.9, 0.4, 0.9, 0.4])
        kps, conf = inner_keypoints_from_corners(
            corners, scores, threshold=0.80,
        )
        self.assertGreater(conf[4], 0.0)
        self.assertEqual(conf[5], 0.0)
        self.assertEqual(kps[5], [0.0, 0.0])
        self.assertGreater(conf[6], 0.0)
        self.assertEqual(conf[7], 0.0)

    def test_all_below_threshold_yields_an_empty_observation(self):
        corners = np.ones((4, 2)) * 50.0
        scores = np.array([0.1, 0.2, 0.1, 0.2])
        kps, conf = inner_keypoints_from_corners(
            corners, scores, threshold=0.80,
        )
        self.assertTrue(all(c == 0.0 for c in conf))
        self.assertTrue(all(p == [0.0, 0.0] for p in kps))

    def test_handoff_decode_order_is_tl_tr_br_bl(self):
        """A peak in channel 0 must decode as TL, matching the README."""
        import sys
        from pathlib import Path

        root = str(Path('gatenet_handoff').resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from aigp.perception.cnn_decode import decode_heatmaps

        hm = np.full((1, 4, 45, 80), -10.0, dtype=np.float32)
        off = np.zeros((1, 8, 45, 80), dtype=np.float32)
        # Channel 0 (TL) wins at cell (10, 20): pixel = (20, 10) * stride 4.
        hm[0, 0, 10, 20] = 5.0
        corners, scores = decode_heatmaps(hm, off, (320, 180))
        self.assertGreater(scores[0, 0], 0.9)
        # normalised: px/320, py/180
        self.assertAlmostEqual(float(corners[0, 0, 0]), 80.0 / 320.0, places=5)
        self.assertAlmostEqual(float(corners[0, 0, 1]), 40.0 / 180.0, places=5)


class PanelGatenetTests(unittest.TestCase):
    def test_panel_draws_confident_inner_corners(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest('cv2 unavailable')
        from obs_panel import ObservationPanel

        panel = ObservationPanel()
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        panel._draw_gatenet(img, {'gatenet': {
            'n_seen': 4,
            'elapsed_ms': 4.2,
            'threshold': 0.80,
            'names': ['TL', 'TR', 'BR', 'BL'],
            'scores': [0.91, 0.88, 0.93, 0.85],
            'corners_px': [
                [120, 90], [280, 90], [280, 250], [120, 250],
            ],
        }})
        self.assertGreater(int(img.sum()), 0)

    def test_render_shows_only_gatenet_when_payload_present(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest('cv2 unavailable')
        from obs_panel import ObservationPanel
        from test_obs_panel import _shared

        shared = _shared()
        shared['gatenet'] = {
            'n_seen': 4,
            'elapsed_ms': 4.2,
            'threshold': 0.80,
            'names': ['TL', 'TR', 'BR', 'BL'],
            'scores': [0.91, 0.88, 0.93, 0.85],
            'corners_px': [
                [120, 90], [280, 90], [280, 250], [120, 250],
            ],
        }
        img = ObservationPanel().render(shared)
        # YOLO kp0 from _shared sits at (100, 150); exclusive GateNet must
        # leave that neighbourhood untouched. GateNet's TL diamond is at (120, 90).
        self.assertEqual(int(img[145:156, 95:106].sum()), 0)
        self.assertGreater(int(img[82:99, 112:129].sum()), 0)

    def test_render_shows_only_yolo_when_no_gatenet(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest('cv2 unavailable')
        from obs_panel import ObservationPanel
        from test_obs_panel import _shared

        shared = _shared()
        img = ObservationPanel().render(shared)
        # YOLO kp0 is a filled circle at (100, 150). No GateNet diamond at
        # a location GateNet would use if it were overlaid.
        self.assertGreater(int(img[150, 100].sum()), 0)
        self.assertEqual(int(img[82:99, 112:129].sum()), 0)

    def test_panel_tolerates_no_gatenet_data(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest('cv2 unavailable')
        from obs_panel import ObservationPanel

        img = np.zeros((360, 640, 3), dtype=np.uint8)
        ObservationPanel()._draw_gatenet(img, {})
        self.assertEqual(int(img.sum()), 0)

    def test_weak_corners_are_drawn_dim_not_as_detections(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest('cv2 unavailable')
        from obs_panel import ObservationPanel

        strong = np.zeros((360, 640, 3), dtype=np.uint8)
        weak = np.zeros((360, 640, 3), dtype=np.uint8)
        payload = {
            'n_seen': 1,
            'threshold': 0.80,
            'names': ['TL', 'TR', 'BR', 'BL'],
            'corners_px': [[120, 90], [280, 90], [280, 250], [120, 250]],
        }
        ObservationPanel()._draw_gatenet(strong, {'gatenet': {
            **payload, 'scores': [0.91, 0.88, 0.93, 0.85],
        }})
        ObservationPanel()._draw_gatenet(weak, {'gatenet': {
            **payload, 'scores': [0.1, 0.1, 0.1, 0.1],
        }})
        self.assertGreater(int(strong.sum()), int(weak.sum()))


class ConfigTests(unittest.TestCase):
    def test_gatenet_config_keys_exist(self):
        import config
        self.assertTrue(hasattr(config, 'GATENET_ENABLED'))
        self.assertTrue(hasattr(config, 'GATENET_SCORE_THRESHOLD'))
        self.assertTrue(hasattr(config, 'GATENET_MODEL_PATH'))

    def test_observe_only_defaults_off(self):
        import config
        self.assertFalse(config.GATENET_ENABLED)


class VisionCompatTests(unittest.TestCase):
    def test_timings_do_not_require_last_debug(self):
        """The attribute YOLO always sets; GateNet originally did not.

        Accessing ``detector.last_debug`` killed the vision thread after the
        first GateNet frame, so the panel froze on that image.
        """
        from vision_rx import _detector_timings

        class Bare:
            pass

        timings = _detector_timings(Bare(), 12.5)
        self.assertEqual(timings['total'], 12.5)

    def test_timings_copy_yolo_debug_when_present(self):
        from vision_rx import _detector_timings

        class Debug:
            timings_ms = {'detect': 3.0}

        class Det:
            last_debug = Debug()

        timings = _detector_timings(Det(), 4.0)
        self.assertEqual(timings['detect'], 3.0)
        self.assertEqual(timings['total'], 4.0)


class LiveOnnxTests(unittest.TestCase):
    """Hit the real weights if onnxruntime and a captured frame are present."""

    def test_onnx_returns_four_corners_on_a_known_gate_frame(self):
        try:
            import onnxruntime  # noqa: F401
            import cv2
        except ImportError:
            self.skipTest('onnxruntime or cv2 missing')
        from pathlib import Path

        from vision.gatenet_detector import DEFAULT_ONNX, GateNetDetector

        if not DEFAULT_ONNX.is_file():
            self.skipTest('gatenet.onnx missing')
        frames = sorted(Path('frames').glob('run_*/*.jpg'))
        if not frames:
            self.skipTest('no captured frames')
        img = cv2.imread(str(frames[0]))
        if img is None:
            self.skipTest('could not read frame')
        det = GateNetDetector(providers=['CPUExecutionProvider'])
        res = det.infer(img)
        self.assertEqual(res.corners_px.shape, (4, 2))
        self.assertIsNotNone(det.last_debug)
        self.assertIn('gatenet', det.last_debug.timings_ms)
        self.assertEqual(res.scores.shape, (4,))
        self.assertGreater(res.elapsed_ms, 0.0)


if __name__ == '__main__':
    unittest.main()
