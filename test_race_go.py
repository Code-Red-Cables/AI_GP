"""Arming must follow the sim's race clock, not a wall-clock guess.

The countdown runs on sim time. Under client slow-mo a wall-clock hold arms
early -- measured at 1.5 s before GO on a real 0.2x run -- which is an early
start, and it also means the policy begins flying before the gates are live.
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'tools'))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    'tf', Path(__file__).resolve().parent / 'tools' / 'tune_flight.py'
)
tf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tf)


class RaceGoTests(unittest.TestCase):
    def test_waits_for_the_published_start_time(self):
        shared = {'race_status': {'sim_boot_ms': 100, 'race_start_ms': -1}}

        def advance():
            time.sleep(0.05)
            shared['race_status'] = {'sim_boot_ms': 200, 'race_start_ms': 3298}
            time.sleep(0.05)
            shared['race_status'] = {'sim_boot_ms': 3300, 'race_start_ms': 3298}

        t = threading.Thread(target=advance, daemon=True)
        t.start()
        self.assertTrue(tf._wait_for_race_go(shared, timeout_s=5.0))
        t.join()

    def test_does_not_fire_on_the_stale_previous_race(self):
        """Before the reset lands the sim reports the last race, long finished."""
        shared = {'race_status': {'sim_boot_ms': 30049, 'race_start_ms': 3298}}
        fired: list = []

        def watch():
            fired.append(tf._wait_for_race_go(shared, timeout_s=0.4))

        t = threading.Thread(target=watch, daemon=True)
        t.start()
        time.sleep(0.15)
        # Still nothing: the stale pair satisfies boot >= start but the reset
        # has not landed, so phase 1 must still be blocking.
        self.assertEqual(fired, [])
        t.join(timeout=2.0)

    def test_reports_failure_when_the_clock_never_appears(self):
        self.assertFalse(tf._wait_for_race_go({}, timeout_s=0.2))

    def test_already_pending_countdown_after_1x_settle_still_reaches_go(self):
        """1x settle often skips the start=-1 blip; boot < start is enough."""
        shared = {'race_status': {'sim_boot_ms': 1500, 'race_start_ms': 3298}}

        def advance():
            time.sleep(0.05)
            shared['race_status'] = {'sim_boot_ms': 3300, 'race_start_ms': 3298}

        threading.Thread(target=advance, daemon=True).start()
        self.assertTrue(tf._wait_for_race_go(shared, timeout_s=2.0))

    def test_on_tick_runs_while_waiting(self):
        shared = {'race_status': {'sim_boot_ms': 100, 'race_start_ms': 500}}
        ticks = []

        def advance():
            time.sleep(0.08)
            shared['race_status'] = {'sim_boot_ms': 600, 'race_start_ms': 500}

        threading.Thread(target=advance, daemon=True).start()
        self.assertTrue(tf._wait_for_race_go(
            shared, timeout_s=2.0, on_tick=lambda: ticks.append(1),
        ))
        self.assertGreater(len(ticks), 0)

    def test_no_pending_countdown_returns_immediately(self):
        shared = {'race_status': {'sim_boot_ms': 100, 'race_start_ms': -1}}

        def advance():
            time.sleep(0.05)
            # Start time already in the past: nothing to wait for.
            shared['race_status'] = {'sim_boot_ms': 9000, 'race_start_ms': 3298}

        threading.Thread(target=advance, daemon=True).start()
        self.assertTrue(tf._wait_for_race_go(shared, timeout_s=3.0))


if __name__ == '__main__':
    unittest.main()
