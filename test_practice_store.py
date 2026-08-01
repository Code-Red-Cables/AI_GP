"""Offline tests for practice attitude-tape checkpoints."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from attitude_tape import AttitudeTapeRecorder
import practice_store as ps


class PracticeStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.patches = [
            mock.patch.object(ps, 'PRACTICE_DIR', self.root),
            mock.patch.object(ps, 'INDEX_PATH', self.root / 'index.json'),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def _rec_with_gates(self, times: dict[int, float]) -> AttitudeTapeRecorder:
        rec = AttitudeTapeRecorder(name='test')
        rec.start(0.0)
        # Fake wall clock via direct samples + gate tags.
        rec._t0 = 0.0
        t_end = max(times.values()) + 0.3
        t = 0.0
        while t <= t_end:
            # Vary pitch a bit so samples aren't identical.
            rec.samples.append({
                't': round(t, 4),
                'des_roll': 0.02 * (1.0 if int(t * 10) % 2 == 0 else -1.0),
                'des_pitch': 0.15,
                'yaw_rate': 0.05,
                'thrust': 0.26,
                'pad_pitch': 0.7,
            })
            t += 0.05
        rec._last_sample_t = rec.samples[-1]['t']
        for g, gt in sorted(times.items()):
            rec.gate_passes.append({'t': float(gt), 'gate': int(g)})
        rec._last_gate = max(times)
        return rec

    def test_saves_faster_through_gate(self):
        rec = self._rec_with_gates({1: 2.0, 2: 4.0})
        msg = ps.maybe_update_through_gate(rec, 2, source='test')
        self.assertIsNotNone(msg)
        self.assertIn('SAVED', msg)
        self.assertTrue(ps.through_path(2).exists())
        self.assertAlmostEqual(ps.load_index()['gates']['2']['time_s'], 4.0)

        slower = self._rec_with_gates({1: 2.0, 2: 5.0})
        msg2 = ps.maybe_update_through_gate(slower, 2, source='test')
        self.assertIn('not faster', msg2 or '')

        faster = self._rec_with_gates({1: 1.8, 2: 3.5})
        msg3 = ps.maybe_update_through_gate(faster, 2, source='test')
        self.assertIn('NEW BEST', msg3 or '')
        self.assertAlmostEqual(ps.load_index()['gates']['2']['time_s'], 3.5)

        loaded = ps.load_through_gate(2)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded['type'], 'attitude_tape')
        self.assertGreater(len(loaded['samples']), 10)
        # Pad axis preserved on samples when present.
        self.assertTrue(
            any('pad_pitch' in s for s in loaded['samples'])
        )

    def test_requires_earlier_gates(self):
        rec = self._rec_with_gates({2: 4.0})
        msg = ps.maybe_update_through_gate(rec, 2, source='test')
        self.assertIn('missing earlier', msg or '')

    def test_format_list_empty(self):
        text = ps.format_list()
        self.assertIn('No practice checkpoints', text)


if __name__ == '__main__':
    unittest.main()
