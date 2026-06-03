"""Telemetry + decision logging for offline tuning (see PLAN.md 8.7 / Task E).

The sim is deterministic, so a per-run JSONL log of telemetry, vision estimates
and the planner's targets makes runs reproducible and diffable while tuning gains.
Runs as its own daemon-style thread following the repo's create_*/get_thread_for_join
convention. Logging is data-only (no human interaction), so it is safe in timed runs.
"""

import json
import os
import threading
import time


def _jsonable(o):
    """Best-effort fallback for json.dumps (numpy scalars, tuples, etc.)."""
    try:
        return float(o)
    except Exception:
        return str(o)


class Logger:

    LOG_KEYS = ('armed', 'attitude', 'position_ned', 'odometry', 'race',
                'vision', 'target', 'last_collision')

    def __init__(self, data, path=None, hz=10):
        self.data = data
        if 'lock' not in self.data:
            self.data['lock'] = threading.RLock()
        self.hz = hz
        self.path = path or os.path.join('logs', f'run_{int(time.time())}.jsonl')
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        self.thread = None
        self.is_running = False

    @classmethod
    def create_logger(cls, data, **kwargs):
        lg = cls(data, **kwargs)
        lg.thread = threading.Thread(target=lg._log_loop, daemon=False)
        lg.is_running = True
        lg.thread.start()
        print(f"Logging run to {lg.path}", flush=True)
        return lg

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def _snapshot(self):
        with self.data['lock']:
            return {k: self.data.get(k) for k in self.LOG_KEYS}

    def _log_loop(self):
        dt = 1.0 / self.hz
        with open(self.path, 'a', buffering=1) as f:
            while self.is_running:
                rec = {'t_ns': time.time_ns()}
                rec.update(self._snapshot())
                try:
                    f.write(json.dumps(rec, default=_jsonable) + '\n')
                except Exception:
                    pass
                time.sleep(dt)
