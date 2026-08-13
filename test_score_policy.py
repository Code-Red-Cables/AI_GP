"""Unit tests for autonomous eval scoring."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from tools.score_policy import score_telem


def _write(rows):
    tmp = tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False, newline='')
    path = Path(tmp.name)
    writer = csv.DictWriter(
        tmp,
        fieldnames=[
            't', 'active_gate', 'control_authority', 'race_finish_ns',
        ],
    )
    writer.writeheader()
    writer.writerows(rows)
    tmp.close()
    return path


class ScorePolicyTests(unittest.TestCase):
    def test_finished_lap(self):
        path = _write([
            {'t': '0', 'active_gate': '0', 'control_authority': 'policy',
             'race_finish_ns': 'nan'},
            {'t': '10', 'active_gate': '17', 'control_authority': 'policy',
             'race_finish_ns': '1.2e10'},
        ])
        r = score_telem(path, finish_gate=17)
        self.assertTrue(r['finished'])
        self.assertEqual(r['gates_cleared'], 17)
        self.assertEqual(r['human_frames'], 0)

    def test_fail_gate(self):
        path = _write([
            {'t': '0', 'active_gate': '0', 'control_authority': 'policy',
             'race_finish_ns': 'nan'},
            {'t': '5', 'active_gate': '3', 'control_authority': 'policy',
             'race_finish_ns': 'nan'},
        ])
        r = score_telem(path, finish_gate=17)
        self.assertFalse(r['finished'])
        self.assertEqual(r['fail_gate'], 3)

    def test_counts_human_frames(self):
        path = _write([
            {'t': '0', 'active_gate': '1', 'control_authority': 'human',
             'race_finish_ns': 'nan'},
            {'t': '1', 'active_gate': '1', 'control_authority': 'policy',
             'race_finish_ns': 'nan'},
        ])
        r = score_telem(path, finish_gate=17)
        self.assertEqual(r['human_frames'], 1)


if __name__ == '__main__':
    unittest.main()
