"""Offline tests for practice attitude-tape checkpoints."""

from __future__ import annotations

import json
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

    def test_save_run_partial_vs_complete(self):
        partial = self._rec_with_gates({1: 2.0, 2: 4.0, 6: 11.0})
        msg = ps.save_run(partial, reason='reset', source='test')
        self.assertIsNotNone(msg)
        self.assertIn('PARTIAL', msg or '')
        self.assertTrue(any(ps.runs_partial_dir().glob('partial_*.json')))
        self.assertFalse(any(ps.runs_complete_dir().glob('complete_*.json')))

        # Complete = cleared course last gate (17), not merely race_finish_ns.
        times = {g: float(g) * 2.0 for g in range(1, 18)}
        complete = self._rec_with_gates(times)
        msg2 = ps.save_run(complete, reason='quit', source='test')
        self.assertIsNotNone(msg2)
        self.assertIn('COMPLETE', msg2 or '')
        saved = list(ps.runs_complete_dir().glob('complete_*.json'))
        self.assertEqual(len(saved), 1)
        tape = json.loads(saved[0].read_text(encoding='utf-8'))
        self.assertEqual(tape['run']['kind'], 'complete')
        self.assertEqual(tape['run']['max_gate'], 17)

        idx = ps.load_runs_index()
        self.assertEqual(len(idx['runs']), 2)
        self.assertEqual(idx['runs'][0]['kind'], 'partial')
        self.assertEqual(idx['runs'][1]['kind'], 'complete')

        text = ps.format_list()
        self.assertIn('1 complete', text)
        self.assertIn('1 partial', text)

    def test_save_run_skips_empty(self):
        rec = AttitudeTapeRecorder(name='empty')
        rec.start(0.0)
        self.assertIsNone(ps.save_run(rec, reason='quit'))


if __name__ == '__main__':
    unittest.main()
