"""The input panel must show what the network gets, and never kill a flight."""
from __future__ import annotations

import unittest

import numpy as np

from obs_panel import PANEL_W, ObservationPanel
from race_obs import FEATURE_DIM, FEATURE_DIM_CTX


def _shared(*, frame=True, keypoints=8, gate=7, ahrs=True):
    kps = [(100.0 + 40 * i, 150.0 + 10 * i) for i in range(keypoints)]
    kps += [(0.0, 0.0)] * (8 - keypoints)
    shared = {
        'gate_detection': {
            'keypoints_px': kps,
            'keypoint_confidences': [0.9] * keypoints + [0.0] * (8 - keypoints),
        },
        'highres_imu': {'xgyro': 0.1, 'ygyro': -0.2, 'zgyro': 0.05},
        'race_status': {'active_gate': gate},
        'planner_target': {
            'thrust': 0.31, 'roll_rate': 1.2,
            'pitch_rate': -0.3, 'yaw_rate': 0.0,
        },
        'planner_mode': 'test',
        'control_authority': 'policy',
    }
    if ahrs:
        shared['control_output'] = {'ahrs_roll': 0.05, 'ahrs_pitch': -0.02}
    if frame:
        shared['debug_frame'] = np.zeros((360, 640, 3), dtype=np.uint8)
    return shared


class PanelTests(unittest.TestCase):
    def test_renders_frame_plus_sidebar(self):
        p = ObservationPanel()
        img = p.render(_shared())
        self.assertEqual(img.shape[1], 640 + PANEL_W)
        # The sidebar is taller than the 360 px frame and must not be clipped;
        # the frame side is padded to match instead.
        self.assertGreaterEqual(img.shape[0], 620)

    def test_survives_a_missing_camera_frame(self):
        p = ObservationPanel()
        img = p.render(_shared(frame=False))
        self.assertIsNotNone(img)
        self.assertEqual(img.shape[1], 640 + PANEL_W)

    def test_context_mode_reads_the_wider_observation(self):
        from policy_planner import observation_from_shared

        shared = _shared()
        self.assertEqual(len(observation_from_shared(shared)), FEATURE_DIM)
        self.assertEqual(
            len(observation_from_shared(shared, with_context=True)),
            FEATURE_DIM_CTX,
        )
        p = ObservationPanel(with_context=True)
        self.assertIsNotNone(p.render(shared))

    def test_partial_keypoints_render(self):
        p = ObservationPanel()
        self.assertIsNotNone(p.render(_shared(keypoints=3)))

    def test_attitude_source_is_reported_honestly(self):
        p = ObservationPanel()
        self.assertEqual(p._attitude_source(_shared(ahrs=True))[0],
                         'controller AHRS')
        self.assertEqual(p._attitude_source(_shared(ahrs=False))[0],
                         'EKF (DRIFTS!)')
        raw = _shared(ahrs=False)
        raw['attitude_raw'] = {'roll': 0.0, 'pitch': 0.0}
        self.assertEqual(p._attitude_source(raw)[0], 'sim ATTITUDE')

    def test_show_never_raises_on_bad_state(self):
        """A HUD failure must disable the panel, not take the flight with it."""
        p = ObservationPanel()
        p._cv2 = object()  # any attribute access will explode
        self.assertFalse(p.show(_shared()))

    def test_race_pose_section_appears_when_present(self):
        shared = _shared()
        shared['race_pose'] = {
            'mode': 'align', 'range_m': 4.2, 'lateral_m': -0.3,
            'vertical_m': 0.1, 'bearing_rad': -0.07, 'residual_m': 0.2,
        }
        self.assertIsNotNone(ObservationPanel().render(shared))


if __name__ == '__main__':
    unittest.main()
