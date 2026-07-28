"""Unit tests for multi-gate bearing latch and post-pass look steering."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from vision.gate_bearings import (
    GateBearingTable,
    GateObservation,
    clamp_contact_vertical,
    near_course_observations,
    observe_pose_candidates,
)
from vision.yolo_gate_detector import YoloGateBox


def _candidate(cx, cy, area=4000.0, conf=0.85, hsv=True, idx=0):
    side = math.sqrt(max(area, 1.0))
    x1, y1 = cx - side * 0.5, cy - side * 0.5
    x2, y2 = cx + side * 0.5, cy + side * 0.5
    box = YoloGateBox(
        bbox=(x1, y1, x2, y2),
        confidence=conf,
        source_index=idx,
    )
    # Degenerate keypoints so observe falls back to image/size ordering.
    keypoints = np.array(
        [[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=np.float32
    )
    return SimpleNamespace(
        box=box,
        keypoints=keypoints,
        hsv_confirmed=hsv,
    )


class GateBearingTests(unittest.TestCase):
    def test_orders_near_to_far_and_latches_upcoming(self):
        # Larger / lower-center box = nearer gate one; smaller / upper-right = gate two.
        near = _candidate(320, 200, area=12000, idx=0)
        mid = _candidate(480, 120, area=3500, idx=1)
        far = _candidate(520, 90, area=1200, idx=2)
        observations = observe_pose_candidates(
            [far, near, mid], 640, 360, require_hsv=True
        )
        self.assertEqual(len(observations), 3)
        self.assertLess(observations[0].range_m, observations[1].range_m)
        self.assertLess(observations[1].range_m, observations[2].range_m)

        table = GateBearingTable()
        nxt = table.update(observations, now=1.0)
        self.assertIsNotNone(nxt)
        self.assertGreater(nxt.horizontal_normalized, 0.0)
        self.assertLess(nxt.vertical_normalized, 0.0)
        # Nearest next (~16 m) is latched; the ~30 m end-course dot is ignored.
        self.assertGreaterEqual(len(table.upcoming), 1)
        self.assertLess(table.upcoming[0].range_m, 25.0)
        self.assertTrue(
            all(b.range_m < 30.0 for b in table.upcoming),
            "far end-of-course gate must not enter the look queue",
        )

        after_pass = table.consume_pass(2.0)
        self.assertIsNotNone(after_pass)
        self.assertGreater(after_pass.horizontal_normalized, 0.0)
        peeked = table.peek_next(2.0)
        self.assertIsNotNone(peeked)
        self.assertAlmostEqual(peeked.range_m, after_pass.range_m, places=3)

    def test_keeps_previous_latch_when_only_one_gate_visible(self):
        table = GateBearingTable()
        table.update(
            [
                GateObservation(
                    range_m=5.0,
                    horizontal_normalized=0.0,
                    vertical_normalized=0.1,
                    yaw_offset_rad=0.0,
                    pitch_offset_rad=0.05,
                    confidence=0.9,
                    source="image",
                    center_xy=(320.0, 200.0),
                ),
                GateObservation(
                    range_m=12.0,
                    horizontal_normalized=0.55,
                    vertical_normalized=-0.35,
                    yaw_offset_rad=0.3,
                    pitch_offset_rad=-0.2,
                    confidence=0.8,
                    source="image",
                    center_xy=(500.0, 110.0),
                ),
            ],
            now=1.0,
        )
        table.update(
            [
                GateObservation(
                    range_m=4.0,
                    horizontal_normalized=0.0,
                    vertical_normalized=0.0,
                    yaw_offset_rad=0.0,
                    pitch_offset_rad=0.0,
                    confidence=0.9,
                    source="image",
                    center_xy=(320.0, 180.0),
                )
            ],
            now=1.5,
        )
        nxt = table.peek_next(1.5)
        self.assertIsNotNone(nxt)
        self.assertAlmostEqual(nxt.horizontal_normalized, 0.55)

    def test_bearing_table_rejects_collapsed_near_gate_refresh(self):
        table = GateBearingTable()
        good = [
            GateObservation(
                range_m=12.0,
                horizontal_normalized=0.0,
                vertical_normalized=0.1,
                yaw_offset_rad=0.0,
                pitch_offset_rad=0.0,
                confidence=0.9,
                source="pnp",
                center_xy=(320.0, 200.0),
            ),
            GateObservation(
                range_m=22.0,
                horizontal_normalized=0.45,
                vertical_normalized=-0.50,
                yaw_offset_rad=0.3,
                pitch_offset_rad=-0.2,
                confidence=0.8,
                source="pnp",
                center_xy=(480.0, 90.0),
            ),
        ]
        table.update(good, now=1.0)
        collapsed = [
            GateObservation(
                range_m=6.0,
                horizontal_normalized=0.0,
                vertical_normalized=0.0,
                yaw_offset_rad=0.0,
                pitch_offset_rad=0.0,
                confidence=0.9,
                source="pnp",
                center_xy=(320.0, 180.0),
            ),
            GateObservation(
                range_m=32.0,
                horizontal_normalized=0.10,
                vertical_normalized=-0.80,
                yaw_offset_rad=0.05,
                pitch_offset_rad=-0.4,
                confidence=0.5,
                source="pnp",
                center_xy=(350.0, 40.0),
            ),
        ]
        table.update(collapsed, now=2.0)
        nxt = table.peek_next(2.0)
        self.assertIsNotNone(nxt)
        # Near-gate freeze or collapse reject must keep the strong +0.45 latch.
        self.assertGreater(nxt.horizontal_normalized, 0.30)

    def test_ignores_far_end_gate_after_pass(self):
        near_next = GateObservation(
            range_m=20.0,
            horizontal_normalized=0.40,
            vertical_normalized=-0.35,
            yaw_offset_rad=0.25,
            pitch_offset_rad=-0.15,
            confidence=0.85,
            source="pnp",
            center_xy=(450.0, 120.0),
        )
        far_end = GateObservation(
            range_m=38.0,
            horizontal_normalized=0.15,
            vertical_normalized=-0.85,
            yaw_offset_rad=0.08,
            pitch_offset_rad=-0.4,
            confidence=0.7,
            source="pnp",
            center_xy=(360.0, 30.0),
        )
        current = GateObservation(
            range_m=8.0,
            horizontal_normalized=0.0,
            vertical_normalized=0.1,
            yaw_offset_rad=0.0,
            pitch_offset_rad=0.0,
            confidence=0.9,
            source="pnp",
            center_xy=(320.0, 200.0),
        )
        near = near_course_observations([current, near_next, far_end])
        self.assertEqual(len(near), 2)
        self.assertAlmostEqual(near[1].range_m, 20.0)


    def test_rejects_through_opening_contact_when_primary_close(self):
        current = GateObservation(
            range_m=2.3,
            horizontal_normalized=0.0,
            vertical_normalized=0.05,
            yaw_offset_rad=0.0,
            pitch_offset_rad=0.0,
            confidence=0.9,
            source="pnp",
            center_xy=(320.0, 190.0),
        )
        end_course = GateObservation(
            range_m=26.5,
            horizontal_normalized=0.39,
            vertical_normalized=-0.56,
            yaw_offset_rad=0.2,
            pitch_offset_rad=-0.25,
            confidence=0.7,
            source="pnp",
            center_xy=(450.0, 80.0),
        )
        near = near_course_observations([current, end_course])
        self.assertEqual(len(near), 1)
        self.assertAlmostEqual(near[0].range_m, 2.3)

        table = GateBearingTable()
        early = [
            GateObservation(
                range_m=10.7,
                horizontal_normalized=0.0,
                vertical_normalized=0.0,
                yaw_offset_rad=0.0,
                pitch_offset_rad=0.0,
                confidence=0.9,
                source="pnp",
                center_xy=(320.0, 180.0),
            ),
            GateObservation(
                range_m=22.9,
                horizontal_normalized=0.34,
                vertical_normalized=-0.30,
                yaw_offset_rad=0.2,
                pitch_offset_rad=-0.15,
                confidence=0.8,
                source="pnp",
                center_xy=(430.0, 120.0),
            ),
        ]
        table.update(early, now=1.0)
        self.assertAlmostEqual(table.live_secondary.range_m, 22.9)
        # Close primary + far through-opening candidate must not steal CONTACT.
        table.update([current, end_course], now=2.0)
        self.assertAlmostEqual(table.live_secondary.range_m, 22.9)

    def test_live_nearest_two_updates_even_when_queue_frozen(self):
        table = GateBearingTable()
        far_pair = [
            GateObservation(
                range_m=12.0,
                horizontal_normalized=0.0,
                vertical_normalized=0.0,
                yaw_offset_rad=0.0,
                pitch_offset_rad=0.0,
                confidence=0.9,
                source="pnp",
                center_xy=(320.0, 180.0),
            ),
            GateObservation(
                range_m=20.0,
                horizontal_normalized=0.45,
                vertical_normalized=-0.30,
                yaw_offset_rad=0.25,
                pitch_offset_rad=-0.15,
                confidence=0.8,
                source="pnp",
                center_xy=(470.0, 120.0),
            ),
        ]
        table.update(far_pair, now=1.0)
        self.assertTrue(table.has_contact_pair)
        self.assertAlmostEqual(table.live_primary.range_m, 12.0)
        self.assertAlmostEqual(table.live_secondary.range_m, 20.0)
        table.freeze()
        moved_contact = [
            GateObservation(
                range_m=6.0,
                horizontal_normalized=0.0,
                vertical_normalized=0.05,
                yaw_offset_rad=0.0,
                pitch_offset_rad=0.0,
                confidence=0.9,
                source="pnp",
                center_xy=(320.0, 190.0),
            ),
            GateObservation(
                range_m=18.0,
                horizontal_normalized=0.60,
                vertical_normalized=-0.25,
                yaw_offset_rad=0.35,
                pitch_offset_rad=-0.12,
                confidence=0.8,
                source="pnp",
                center_xy=(520.0, 130.0),
            ),
        ]
        table.update(moved_contact, now=2.0)
        # Queue may stay frozen, but live contact must track the second gate.
        contact = table.contact_secondary_bearing()
        self.assertIsNotNone(contact)
        self.assertAlmostEqual(contact.horizontal_normalized, 0.60)
        self.assertAlmostEqual(table.live_primary.range_m, 6.0)

    def test_consume_pass_keeps_near_look_not_gate_after_next(self):
        table = GateBearingTable()
        observations = [
            GateObservation(
                range_m=10.0,
                horizontal_normalized=0.0,
                vertical_normalized=0.0,
                yaw_offset_rad=0.0,
                pitch_offset_rad=0.0,
                confidence=0.9,
                source="pnp",
                center_xy=(320.0, 180.0),
            ),
            GateObservation(
                range_m=18.0,
                horizontal_normalized=0.50,
                vertical_normalized=-0.30,
                yaw_offset_rad=0.3,
                pitch_offset_rad=-0.15,
                confidence=0.85,
                source="pnp",
                center_xy=(480.0, 120.0),
            ),
            GateObservation(
                range_m=26.0,
                horizontal_normalized=-0.20,
                vertical_normalized=-0.40,
                yaw_offset_rad=-0.1,
                pitch_offset_rad=-0.2,
                confidence=0.7,
                source="pnp",
                center_xy=(260.0, 100.0),
            ),
        ]
        table.update(observations, now=1.0)
        self.assertEqual(len(table.upcoming), 2)
        target = table.consume_pass(now=2.0)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target.range_m, 18.0)
        # peek must still return the near next gate we are acquiring, not 26m.
        peeked = table.peek_next(2.0)
        self.assertIsNotNone(peeked)
        self.assertAlmostEqual(peeked.range_m, 18.0)
        self.assertAlmostEqual(peeked.horizontal_normalized, 0.50)
        self.assertAlmostEqual(table.expected_next_range_m, 18.0)


if __name__ == "__main__":
    unittest.main()
