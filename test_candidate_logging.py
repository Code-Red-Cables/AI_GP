"""Raw detector candidates must be logged and drawn even when rejected.

Pins the diagnosis gap: a frame where YOLO found the gate and vision_rx's
selection discarded it used to look identical in both the log and the panel to a
frame with nothing in view.
"""
from __future__ import annotations

import unittest

from logger import Logger


def _logger():
    return Logger.__new__(Logger)


class CandidateConfidenceTests(unittest.TestCase):
    def test_best_confidence_of_several(self):
        log = _logger()
        cands = {'items': [
            {'confidence': 0.42}, {'confidence': 0.81}, {'confidence': 0.55},
        ]}
        self.assertEqual(log._cand_best_conf(cands), '0.810')

    def test_no_candidates_is_nan(self):
        self.assertEqual(_logger()._cand_best_conf({}), 'nan')
        self.assertEqual(_logger()._cand_best_conf({'items': []}), 'nan')

    def test_unparseable_confidence_ignored(self):
        cands = {'items': [{'confidence': None}, {'confidence': 'x'},
                           {'confidence': 0.3}]}
        self.assertEqual(_logger()._cand_best_conf(cands), '0.300')

    def test_all_unparseable_is_nan(self):
        cands = {'items': [{'confidence': None}, {}]}
        self.assertEqual(_logger()._cand_best_conf(cands), 'nan')


class PanelCandidateTests(unittest.TestCase):
    def setUp(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest('cv2 unavailable')
        import numpy as np

        from obs_panel import ObservationPanel
        self.np = np
        self.panel = ObservationPanel()

    def _img(self):
        return self.np.zeros((360, 640, 3), dtype=self.np.uint8)

    def test_rejected_box_is_drawn(self):
        """Candidate present, no matching detection -> something is drawn."""
        img = self._img()
        self.panel._draw_candidates(img, {
            'gate_candidates': {
                'frame_id': 7,
                'items': [{'bbox_px': (100, 80, 300, 260), 'confidence': 0.77}],
            },
            'gate_detection': None,
        })
        self.assertGreater(int(img.sum()), 0, 'rejected box left no marking')

    def test_no_candidates_draws_nothing(self):
        img = self._img()
        self.panel._draw_candidates(img, {'gate_candidates': {'items': []}})
        self.assertEqual(int(img.sum()), 0)

    def test_malformed_bbox_draws_no_box(self):
        """Unusable boxes must not be drawn, though the count is still shown."""
        img = self._img()
        self.panel._draw_candidates(img, {
            'gate_candidates': {'items': [{'bbox_px': (1, 2)},
                                          {'bbox_px': None}]},
            'gate_detection': None,
        })
        # The banner occupies the top ~60 rows; nothing should appear below it.
        self.assertEqual(int(img[60:].sum()), 0)

    def test_delivered_and_rejected_look_different(self):
        cand = {
            'frame_id': 7,
            'items': [{'bbox_px': (100, 80, 300, 260), 'confidence': 0.77}],
        }
        rejected = self._img()
        self.panel._draw_candidates(rejected, {
            'gate_candidates': cand, 'gate_detection': None,
        })
        delivered = self._img()
        self.panel._draw_candidates(delivered, {
            'gate_candidates': cand, 'gate_detection': {'frame_id': 7},
        })
        self.assertFalse(
            bool((rejected == delivered).all()),
            'a rejected box is indistinguishable from a delivered one',
        )

    def test_missing_keys_do_not_raise(self):
        self.panel._draw_candidates(self._img(), {})


if __name__ == '__main__':
    unittest.main()
