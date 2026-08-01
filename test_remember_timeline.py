"""Offline tests for key-press remember / replay."""
from __future__ import annotations

import os
import tempfile
import unittest

from remember_timeline import (
    KeyReplayClock,
    KeyTimeline,
    apply_keys_to_hold_state,
    load_timeline,
)


class TestKeyTimeline(unittest.TestCase):
    def test_note_keys_emits_down_up(self):
        tl = KeyTimeline('x')
        ev = tl.note_keys(0.0, {'w'})
        self.assertEqual(len(ev), 1)
        self.assertTrue(ev[0]['down'])
        self.assertEqual(ev[0]['key'], 'w')
        ev = tl.note_keys(1.5, {'w'})
        self.assertEqual(ev, [])
        ev = tl.note_keys(2.0, set())
        self.assertEqual(len(ev), 1)
        self.assertFalse(ev[0]['down'])
        self.assertEqual(tl.keys_held_at(1.0), {'w'})
        self.assertEqual(tl.keys_held_at(2.1), set())

    def test_trim_until_gate_and_roundtrip(self):
        path = os.path.join(tempfile.mkdtemp(), 'keys.json')
        tl = KeyTimeline(path)
        tl.note_keys(0.0, {'w'})
        tl.note_keys(1.0, {'w'})
        tl.events.append(
            {'t': 1.2, 'event': 'gate_pass', 'active_gate': 1, 'name': 'gate1'}
        )
        tl.note_keys(1.5, set())
        tl.note_keys(2.0, {'d'})  # still inside post-gate trim pad
        tl.note_keys(2.5, set())
        tl.note_keys(8.0, {'a'})  # well after pad — must be cut
        tl.note_keys(8.5, set())
        saved = tl.save()
        self.assertIsNotNone(saved)
        loaded = load_timeline(str(saved))
        self.assertIsNotNone(loaded)
        trimmed, warn = loaded.trim_until_gate(1)
        self.assertIsNone(warn)
        self.assertIn('d', trimmed.keys_held_at(2.2))
        self.assertNotIn('a', trimmed.keys_held_at(8.2))
        self.assertIn('w', trimmed.keys_held_at(0.5))

    def test_trim_without_tag_plays_full(self):
        tl = KeyTimeline('x')
        tl.note_keys(0.0, {'w'})
        tl.note_keys(1.0, set())
        full, warn = tl.trim_until_gate(1)
        self.assertIsNotNone(warn)
        self.assertGreaterEqual(len(full), 2)

    def test_apply_keys_sets_pitch(self):
        hold = {}
        apply_keys_to_hold_state(
            {'w'}, hold, lean_rad=0.2, yaw_rate_cmd=0.5,
            thrust_step=0.1, now=10.0,
        )
        self.assertGreater(abs(hold.get('pitch', 0.0)), 0.1)
        self.assertAlmostEqual(hold.get('pitch_t'), 10.0)

    def test_apply_keys_f_uses_faster_sink(self):
        hold = {}
        apply_keys_to_hold_state(
            {'f'}, hold, lean_rad=0.2, yaw_rate_cmd=0.5,
            thrust_step=0.6, now=10.0, sink_step=1.8,
        )
        self.assertAlmostEqual(hold.get('thrust'), -1.8)
        hold2 = {}
        apply_keys_to_hold_state(
            {'f'}, hold2, lean_rad=0.2, yaw_rate_cmd=0.5,
            thrust_step=0.6, now=10.0, sink_step=0.6,
        )
        self.assertAlmostEqual(hold2.get('thrust'), -0.6)

    def test_clip_at_allows_append(self):
        tl = KeyTimeline('x')
        tl.note_keys(0.0, {'w', 'f'})
        tl.note_keys(1.0, {'w', 'f'})
        tl.events.append(
            {'t': 1.2, 'event': 'gate_pass', 'active_gate': 2, 'name': 'gate2'}
        )
        tl.note_keys(5.0, {'w'})
        tl.note_keys(6.0, set())
        tl.clip_at(1.5)
        self.assertLessEqual(max(float(e['t']) for e in tl.events), 1.5 + 1e-6)
        ev = tl.note_keys(0.1, {'w', 'd'})
        self.assertTrue(any(e.get('key') == 'd' and e.get('down') for e in ev))
        self.assertAlmostEqual(ev[0]['t'], 1.6, places=3)

    def test_replay_clock_finishes(self):
        tl = KeyTimeline('x')
        tl.note_keys(0.0, {'w'})
        tl.note_keys(0.1, set())
        clock = KeyReplayClock(tl)
        import time as _time
        clock._t0 = _time.monotonic() - 10.0
        _, held, finished = clock.tick()
        self.assertTrue(finished)
        self.assertEqual(held, set())

    def test_ensure_post_gate_yaw_key_inserts_e(self):
        tl = KeyTimeline('x')
        tl.note_keys(0.0, {'w'})
        tl.note_keys(1.0, set())
        tl.events.append(
            {'t': 1.2, 'event': 'gate_pass', 'active_gate': 1, 'name': 'gate1'}
        )
        self.assertTrue(
            tl.ensure_post_gate_yaw_key(
                1, key='e', yaw_deg=20.0, yaw_rate_deg=40.0, lead_s=0.45,
            )
        )
        # Starts 0.45s before gate (t=0.75); 20°/40°/s = 0.5s hold.
        self.assertIn('e', tl.keys_held_at(0.8))
        self.assertIn('e', tl.keys_held_at(1.2))
        self.assertNotIn('e', tl.keys_held_at(1.4))
        # Re-call refreshes synthetic timing (still returns True).
        self.assertTrue(
            tl.ensure_post_gate_yaw_key(
                1, key='e', yaw_deg=20.0, yaw_rate_deg=40.0, lead_s=0.45,
            )
        )
        trimmed, _ = tl.trim_until_gate(1)
        self.assertTrue(
            trimmed.ensure_post_gate_yaw_key(
                1, key='e', yaw_deg=20.0, yaw_rate_deg=40.0, lead_s=0.45,
            )
        )
        self.assertIn('e', trimmed.keys_held_at(0.8))


if __name__ == '__main__':
    unittest.main()
