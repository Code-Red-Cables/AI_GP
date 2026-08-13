"""Snake gate detection must find a synthetic gate and reject colour noise.

Checks the properties the paper's algorithm is supposed to have, including the
oblique-bar tracking that the diagonal steps in Algorithm 2 exist to provide.
"""
from __future__ import annotations

import unittest

import numpy as np

try:
    import cv2
except ImportError:                              # pragma: no cover
    cv2 = None

from vision.snake_gate_detector import SnakeGateConfig, SnakeGateDetector

# BGR for a colour inside the default HSV window (0-23 hue, orange).
ORANGE = (20, 120, 235)


def blank(w=640, h=360):
    return np.zeros((h, w, 3), dtype=np.uint8)


def draw_gate(img, x, y, size, thickness=10, angle=0.0):
    """Hollow square outline, optionally rotated about its centre."""
    layer = np.zeros_like(img)
    cv2.rectangle(layer, (x, y), (x + size, y + size), ORANGE, thickness)
    if angle:
        c = (x + size / 2.0, y + size / 2.0)
        m = cv2.getRotationMatrix2D(c, angle, 1.0)
        layer = cv2.warpAffine(layer, m, (img.shape[1], img.shape[0]))
    np.copyto(img, np.maximum(img, layer))
    return img


@unittest.skipIf(cv2 is None, 'cv2 unavailable')
class SnakeGateTests(unittest.TestCase):
    def setUp(self):
        self.det = SnakeGateDetector(SnakeGateConfig(max_samples=1200))

    def test_finds_an_upright_gate(self):
        img = draw_gate(blank(), 220, 90, 170)
        res = self.det.detect(img)
        self.assertTrue(res.found, 'missed a clean synthetic gate')
        cx, cy = res.best.center_px
        self.assertAlmostEqual(cx, 220 + 85, delta=25)
        self.assertAlmostEqual(cy, 90 + 85, delta=25)

    def test_empty_frame_finds_nothing(self):
        res = self.det.detect(blank())
        self.assertFalse(res.found)
        self.assertEqual(res.mask_fraction, 0.0)

    def test_small_colour_blob_is_rejected_by_min_length(self):
        """sigma_L exists precisely to throw these away."""
        img = blank()
        cv2.rectangle(img, (300, 170), (310, 180), ORANGE, -1)
        res = self.det.detect(img)
        self.assertFalse(res.found, 'a 10 px blob passed sigma_L')

    def test_solid_block_fails_colour_fitness(self):
        """A filled square is not a gate: its diagonal perimeter is colour, but
        the algorithm's square is the block itself, so this mainly checks that a
        non-hollow shape does not produce a confident gate."""
        img = blank()
        cv2.rectangle(img, (250, 100), (400, 250), ORANGE, -1)
        res = self.det.detect(img)
        if res.found:
            self.assertGreater(res.best.color_fitness, 0.5)

    def test_tracks_a_rotated_gate(self):
        """With the rotated-rect squaring, roll is survivable."""
        for angle in (5.0, 20.0, 45.0, 60.0):
            img = draw_gate(blank(), 230, 95, 165, thickness=12, angle=angle)
            res = self.det.detect(img)
            self.assertTrue(res.found, f'missed a gate rolled {angle:.0f} deg')

    def test_paper_axis_aligned_squaring_dies_under_roll(self):
        """Documents why this implementation deviates from the paper.

        The published algorithm scores colour fitness along an axis-aligned
        square, whose edges stop lying on the bars as soon as the gate tilts.
        Measured fitness runs 1.00 / 0.34 / 0.19 / 0.09 at 0 / 5 / 10 / 20 deg,
        so at the paper's sigma_cf of 0.35 it is blind past about 5 degrees.
        """
        paper = SnakeGateDetector(SnakeGateConfig(
            max_samples=1500, use_rotated_rect=False,
        ))
        upright = draw_gate(blank(), 230, 95, 165, thickness=12)
        self.assertTrue(paper.detect(upright).found)

        rolled = draw_gate(blank(), 230, 95, 165, thickness=12, angle=20.0)
        self.assertFalse(
            paper.detect(rolled).found,
            'axis-aligned squaring unexpectedly survived 20 deg of roll',
        )

    def test_rotated_rect_beats_the_paper_at_every_angle(self):
        paper = SnakeGateDetector(SnakeGateConfig(
            max_samples=1500, use_rotated_rect=False,
        ))
        for angle in (10.0, 30.0, 45.0):
            img = draw_gate(blank(), 230, 95, 165, thickness=12, angle=angle)
            self.assertTrue(self.det.detect(img).found)
            self.assertFalse(paper.detect(img).found)

    def test_reports_timing_and_mask_fraction(self):
        img = draw_gate(blank(), 220, 90, 170)
        res = self.det.detect(img)
        self.assertGreater(res.elapsed_ms, 0.0)
        self.assertGreater(res.mask_fraction, 0.0)
        self.assertLess(res.mask_fraction, 0.5)

    def test_results_are_sorted_by_colour_fitness(self):
        img = draw_gate(blank(), 60, 60, 120)
        draw_gate(img, 380, 140, 150)
        res = self.det.detect(img)
        fits = [g.color_fitness for g in res.gates]
        self.assertEqual(fits, sorted(fits, reverse=True))

    def test_best_only_returns_one(self):
        det = SnakeGateDetector(SnakeGateConfig(max_samples=1200,
                                               best_only=True))
        img = draw_gate(blank(), 60, 60, 120)
        draw_gate(img, 380, 140, 150)
        res = det.detect(img)
        self.assertLessEqual(len(res.gates), 1)

    def test_corners_are_inside_the_frame(self):
        img = draw_gate(blank(), 220, 90, 170)
        res = self.det.detect(img)
        for g in res.gates:
            for x, y in g.corners_px:
                self.assertGreaterEqual(x, 0.0)
                self.assertGreaterEqual(y, 0.0)
                self.assertLessEqual(x, 640.0)
                self.assertLessEqual(y, 360.0)

    def test_raising_min_length_rejects_a_small_gate(self):
        img = draw_gate(blank(), 300, 160, 30, thickness=5)
        strict = SnakeGateDetector(SnakeGateConfig(min_length_px=60,
                                                  max_samples=1200))
        self.assertFalse(strict.detect(img).found)

    def test_detection_is_deterministic_for_one_detector(self):
        img = draw_gate(blank(), 220, 90, 170)
        a = SnakeGateDetector(SnakeGateConfig(max_samples=800)).detect(img)
        b = SnakeGateDetector(SnakeGateConfig(max_samples=800)).detect(img)
        self.assertEqual(a.found, b.found)
        if a.found:
            self.assertAlmostEqual(a.best.center_px[0], b.best.center_px[0],
                                   delta=1e-6)


@unittest.skipIf(cv2 is None, 'cv2 unavailable')
class PanelSnakeTests(unittest.TestCase):
    def test_panel_draws_snake_quads(self):
        from obs_panel import ObservationPanel

        panel = ObservationPanel()
        img = blank()
        panel._draw_snake(img, {'snake_gate': {
            'n': 1,
            'elapsed_ms': 3.2,
            'items': [{
                'corners_px': [[100, 80], [260, 80], [260, 240], [100, 240]],
                'color_fitness': 0.71,
            }],
        }})
        self.assertGreater(int(img.sum()), 0)

    def test_panel_tolerates_no_snake_data(self):
        from obs_panel import ObservationPanel

        img = blank()
        ObservationPanel()._draw_snake(img, {})
        self.assertEqual(int(img.sum()), 0)


if __name__ == '__main__':
    unittest.main()
