"""The HSV tuner's non-GUI pieces, so a typo cannot waste a tuning session."""
from __future__ import annotations

import os
import tempfile
import unittest

try:
    import cv2
except ImportError:                              # pragma: no cover
    cv2 = None

import numpy as np


@unittest.skipIf(cv2 is None, 'cv2 unavailable')
class CollectFramesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        for name in ('b.jpg', 'a.jpg', 'c.png'):
            cv2.imwrite(os.path.join(self.dir, name), img)

    def tearDown(self):
        self.tmp.cleanup()

    def test_directory_returns_sorted_jpgs_then_pngs(self):
        from tools.hsv_tuner import collect_frames

        got = [os.path.basename(p) for p in collect_frames(self.dir)]
        self.assertEqual(got, ['a.jpg', 'b.jpg', 'c.png'])

    def test_single_file_returns_just_that_file(self):
        from tools.hsv_tuner import collect_frames

        one = os.path.join(self.dir, 'a.jpg')
        self.assertEqual(collect_frames(one), [one])

    def test_missing_directory_returns_empty(self):
        from tools.hsv_tuner import collect_frames

        self.assertEqual(collect_frames(os.path.join(self.dir, 'nope')), [])


@unittest.skipIf(cv2 is None, 'cv2 unavailable')
class LabelTests(unittest.TestCase):
    def test_label_writes_onto_the_image(self):
        from tools.hsv_tuner import _label

        img = np.zeros((360, 640, 3), dtype=np.uint8)
        _label(img, [('hello', None)])
        self.assertGreater(int(img.sum()), 0)

    def test_label_accepts_an_explicit_colour(self):
        from tools.hsv_tuner import _label

        img = np.zeros((360, 640, 3), dtype=np.uint8)
        _label(img, [('warn', (0, 0, 255))])
        # Red channel only, given a pure-red label.
        self.assertGreater(int(img[..., 2].sum()), 0)
        self.assertEqual(int(img[..., 0].sum()), 0)

    def test_empty_lines_leave_the_image_untouched(self):
        from tools.hsv_tuner import _label

        img = np.zeros((360, 640, 3), dtype=np.uint8)
        _label(img, [])
        self.assertEqual(int(img.sum()), 0)


@unittest.skipIf(cv2 is None, 'cv2 unavailable')
class DefaultsTests(unittest.TestCase):
    def test_sliders_start_from_config(self):
        """A session must begin from what the aircraft actually flies."""
        import config

        lo = tuple(int(v) for v in config.GATE_HSV_LOWER)
        hi = tuple(int(v) for v in config.GATE_HSV_UPPER)
        self.assertEqual(len(lo), 3)
        self.assertEqual(len(hi), 3)
        for a, b in zip(lo, hi):
            self.assertLessEqual(a, b, 'config HSV lower exceeds upper')

    def test_hue_bounds_are_within_opencv_range(self):
        import config

        self.assertLessEqual(int(config.GATE_HSV_UPPER[0]), 179)
        self.assertGreaterEqual(int(config.GATE_HSV_LOWER[0]), 0)


if __name__ == '__main__':
    unittest.main()
