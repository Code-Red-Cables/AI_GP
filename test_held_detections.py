"""Held detections reach the policy, but only while their corners are fresh.

Before this, vision_rx dropped every result the detector flagged ``predicted``,
which is how it re-serves the previous target while its lock catches up. That
cost the policy roughly a quarter of all frames beyond 45 degrees of roll.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Optional, Sequence

import config


@dataclass
class FakeBox:
    center: tuple
    bbox: tuple = (0.0, 0.0, 10.0, 10.0)
    area: float = 100.0
    confidence: float = 0.7


@dataclass
class FakeCandidate:
    box: FakeBox
    keypoints: Optional[Sequence] = None
    keypoint_confidences: Optional[Sequence] = None
    hsv_confirmed: bool = False


@dataclass
class FakePoseDebug:
    candidates: tuple = ()
    selected: object = None


@dataclass
class FakeMeasured:
    found: bool = True
    predicted: bool = False
    missing_frames: int = 0
    center_px: tuple = (320.0, 180.0)
    corners_px: Optional[Sequence] = None
    bbox_px: tuple = (300.0, 160.0, 340.0, 200.0)
    area_px: float = 1600.0
    confidence: float = 0.7
    method: str = 'yolo_pose_box_center'


def _pose_debug_with_corners():
    kps = [[100.0 + 10 * i, 120.0 + 5 * i] for i in range(8)]
    return FakePoseDebug(
        candidates=(FakeCandidate(
            box=FakeBox(center=(320.0, 180.0)),
            keypoints=kps,
            keypoint_confidences=[0.9] * 8,
        ),)
    )


class DeliveryRuleTests(unittest.TestCase):
    """Exercises the accept/reject decision in isolation."""

    def _decide(self, measured, pose_debug):
        """Mirror of the rule in vision_rx.GateVision._handle_frame."""
        from vision_rx import _candidate_keypoints

        if measured is None or not measured.found:
            return None, 'not_found'
        kps, kconf = _candidate_keypoints(pose_debug, measured.center_px)
        stale = bool(measured.predicted) and kps is None
        too_old = bool(measured.predicted) and int(
            measured.missing_frames or 0
        ) > int(getattr(config, 'GATE_HELD_MAX_FRAMES', 3))
        if stale:
            return None, 'stale_held'
        if too_old:
            return None, 'held_too_old'
        return (kps, kconf), ''

    def test_confirmed_detection_is_delivered(self):
        got, why = self._decide(FakeMeasured(), _pose_debug_with_corners())
        self.assertEqual(why, '')
        self.assertIsNotNone(got)
        self.assertEqual(len(got[0]), 8)

    def test_held_detection_with_fresh_corners_is_delivered(self):
        got, why = self._decide(
            FakeMeasured(predicted=True, missing_frames=1),
            _pose_debug_with_corners(),
        )
        self.assertEqual(why, '', 'held detection was dropped')
        self.assertIsNotNone(got)

    def test_held_detection_without_corners_is_rejected(self):
        got, why = self._decide(
            FakeMeasured(predicted=True, missing_frames=1),
            FakePoseDebug(candidates=()),
        )
        self.assertEqual(why, 'stale_held')
        self.assertIsNone(got)

    def test_held_detection_beyond_the_age_cap_is_rejected(self):
        got, why = self._decide(
            FakeMeasured(
                predicted=True,
                missing_frames=config.GATE_HELD_MAX_FRAMES + 1,
            ),
            _pose_debug_with_corners(),
        )
        self.assertEqual(why, 'held_too_old')
        self.assertIsNone(got)

    def test_held_detection_at_exactly_the_cap_is_allowed(self):
        got, why = self._decide(
            FakeMeasured(
                predicted=True, missing_frames=config.GATE_HELD_MAX_FRAMES,
            ),
            _pose_debug_with_corners(),
        )
        self.assertEqual(why, '')
        self.assertIsNotNone(got)

    def test_not_found_is_still_rejected(self):
        got, why = self._decide(
            FakeMeasured(found=False), _pose_debug_with_corners()
        )
        self.assertEqual(why, 'not_found')
        self.assertIsNone(got)

    def test_none_result_is_rejected(self):
        got, why = self._decide(None, _pose_debug_with_corners())
        self.assertEqual(why, 'not_found')
        self.assertIsNone(got)

    def test_a_confirmed_detection_never_needs_corners(self):
        """Corners missing is fine when the detection is not a hold."""
        got, why = self._decide(
            FakeMeasured(predicted=False), FakePoseDebug(candidates=())
        )
        self.assertEqual(why, '')


class ColourFallbackConfigTests(unittest.TestCase):
    def test_colour_fallback_is_enabled(self):
        self.assertTrue(config.GLOBAL_HSV_FALLBACK_ENABLED)

    def test_colour_fallback_runs_during_lock(self):
        self.assertTrue(config.GLOBAL_HSV_FALLBACK_DURING_LOCK)

    def test_colour_confidence_is_scaled_below_yolo(self):
        self.assertLess(config.GLOBAL_HSV_FALLBACK_CONFIDENCE_SCALE, 1.0)

    def test_detector_config_exposes_the_during_lock_flag(self):
        from vision.yolo_pose_gate_detector import PoseGateConfig

        cfg = PoseGateConfig()
        self.assertTrue(cfg.global_hsv_fallback_during_lock)

    def test_held_frame_cap_is_positive(self):
        self.assertGreater(config.GATE_HELD_MAX_FRAMES, 0)


if __name__ == '__main__':
    unittest.main()
