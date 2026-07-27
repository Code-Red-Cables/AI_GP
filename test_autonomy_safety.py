"""Safety-boundary tests for vision commands entering the flight controller."""

import math
import time
import unittest

from opencv_gate_planner import OpenCVGatePlanner


class PlannerSafetyTests(unittest.TestCase):
    def _shared(self, **navigation):
        events = []
        return {
            "navigation": {
                "ts": time.time_ns(),
                "forward_mps": 0.2,
                "right_mps": 0.0,
                "down_mps": 0.0,
                "yaw_rate_rps": 0.0,
                "state": "ALIGN_AND_APPROACH",
                **navigation,
            },
            "attitude": {"yaw": 0.0},
            "log_event": lambda event, detail="": events.append(
                (event, detail)
            ),
            "_events": events,
        }

    def test_nonfinite_navigation_commands_safe_hover(self):
        planner = OpenCVGatePlanner()
        shared = self._shared(forward_mps=math.nan)

        target = planner.compute_target(shared)

        self.assertEqual(
            target,
            {"vn": 0.0, "ve": 0.0, "vd": 0.0, "yaw_rate": 0.0},
        )
        self.assertEqual(shared["planner_mode"], "opencv_stale")
        self.assertEqual(shared["_events"][-1][0], "PLANNER_SAFETY")

    def test_future_navigation_timestamp_safe_hover(self):
        planner = OpenCVGatePlanner()
        shared = self._shared(ts=time.time_ns() + 1_000_000_000)

        target = planner.compute_target(shared)

        self.assertEqual(target["vn"], 0.0)
        self.assertIn("future_navigation", shared["_events"][-1][1])

    def test_valid_body_command_is_rotated_to_ned(self):
        planner = OpenCVGatePlanner()
        shared = self._shared(forward_mps=0.4, right_mps=0.2)
        shared["attitude"]["yaw"] = math.pi / 2.0

        target = planner.compute_target(shared)

        self.assertAlmostEqual(target["vn"], -0.2)
        self.assertAlmostEqual(target["ve"], 0.4)
        self.assertEqual(shared["planner_mode"], "opencv_align_and_approach")


if __name__ == "__main__":
    unittest.main()
