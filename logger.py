"""CSV telemetry + event log for keyboard teleop."""
import csv
import math
import os
import threading
import time
from datetime import datetime

LOG_DIR = 'logs'
LOG_HZ = 50


class Logger:
    def __init__(self, shared_data):
        self.data = shared_data
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(LOG_DIR, exist_ok=True)
        self._csv_path = os.path.join(LOG_DIR, f'telem_{ts}.csv')
        self._evt_path = os.path.join(LOG_DIR, f'events_{ts}.txt')
        self._t0 = time.time()
        self._csv_file = None
        self._csv_writer = None
        self._running = True
        shared_data['log_event'] = self.log_event
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f'[LOG] telem  → {self._csv_path}', flush=True)
        print(f'[LOG] events → {self._evt_path}', flush=True)

    def log_event(self, event: str, detail: str = '') -> None:
        t = time.time() - self._t0
        line = f'{t:9.3f}  {event:<30s}  {detail}'
        print(f'[EVT] {line}', flush=True)
        with open(self._evt_path, 'a') as f:
            f.write(line + '\n')

    def stop(self) -> None:
        self._running = False
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)

    def _loop(self):
        interval = 1.0 / LOG_HZ
        while self._running:
            t0 = time.time()
            try:
                self._write_row()
            except Exception as exc:
                print(f'[LOG] write error: {exc}', flush=True)
            sleep = interval - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
        if self._csv_file:
            self._csv_file.close()

    def _f(self, v, prec: int = 4) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return 'nan'
        return f'{v:.{prec}f}'

    def _write_row(self):
        d = self.data
        att = d.get('attitude') or {}
        imu = d.get('highres_imu') or {}
        ctrl = d.get('control_output') or {}
        tel = d.get('teleop_cmd') or {}
        tgt = d.get('planner_target') or {}
        race = d.get('race_status') or {}

        row = {
            't': self._f(time.time() - self._t0),
            'roll': self._f(att.get('roll')),
            'pitch': self._f(att.get('pitch')),
            'yaw': self._f(att.get('yaw')),
            'ax': self._f(imu.get('xacc')),
            'ay': self._f(imu.get('yacc')),
            'az': self._f(imu.get('zacc')),
            'gx': self._f(imu.get('xgyro')),
            'gy': self._f(imu.get('ygyro')),
            'gz': self._f(imu.get('zgyro')),
            'cmd_thrust': self._f(ctrl.get('thrust')),
            'cmd_roll_rate': self._f(ctrl.get('roll_rate')),
            'cmd_pitch_rate': self._f(ctrl.get('pitch_rate')),
            'cmd_yaw_rate': self._f(ctrl.get('yaw_rate')),
            'ahrs_roll': self._f(ctrl.get('ahrs_roll')),
            'ahrs_pitch': self._f(ctrl.get('ahrs_pitch')),
            'tgt_vn': self._f(tgt.get('vn')),
            'tgt_ve': self._f(tgt.get('ve')),
            'tgt_vd': self._f(tgt.get('vd')),
            'tgt_yaw_rate': self._f(tgt.get('yaw_rate')),
            'teleop_fwd': str(tel.get('fwd', '')),
            'teleop_right': str(tel.get('right', '')),
            'teleop_up': str(tel.get('up', '')),
            'teleop_yaw': str(tel.get('yaw', '')),
            'active_gate': str(race.get('active_gate', '')),
            'safety': str(ctrl.get('safety_reason') or ''),
        }

        if self._csv_writer is None:
            self._csv_file = open(self._csv_path, 'w', newline='')
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(row.keys())
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()
