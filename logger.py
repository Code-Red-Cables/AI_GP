import csv
import math
import os
import threading
import time
from datetime import datetime

LOG_DIR = 'logs'
LOG_HZ  = 50

_NAN = float('nan')


class Logger:
    """
    Background thread that writes a CSV telemetry row at LOG_HZ and an
    events text file on demand.  Call log_event() from any thread.
    shared_data['log_event'] is set to this method so other components
    can log without importing this module.
    """

    def __init__(self, shared_data):
        self.data = shared_data
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(LOG_DIR, exist_ok=True)
        self._csv_path = os.path.join(LOG_DIR, f'telem_{ts}.csv')
        self._evt_path = os.path.join(LOG_DIR, f'events_{ts}.txt')
        self._t0 = time.time()
        self._csv_file   = None
        self._csv_writer = None
        self._running = True
        shared_data['log_event'] = self.log_event
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f'[LOG] telem  → {self._csv_path}', flush=True)
        print(f'[LOG] events → {self._evt_path}', flush=True)

    # ------------------------------------------------------------------
    def log_event(self, event: str, detail: str = '') -> None:
        t = time.time() - self._t0
        line = f'{t:9.3f}  {event:<30s}  {detail}'
        print(f'[EVT] {line}', flush=True)
        with open(self._evt_path, 'a') as f:
            f.write(line + '\n')

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    def _loop(self):
        interval = 1.0 / LOG_HZ
        while self._running:
            t0 = time.time()
            try:
                self._write_row()
            except Exception as e:
                print(f'[LOG] write error: {e}', flush=True)
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
        d    = self.data
        att  = d.get('attitude')        or {}
        imu  = d.get('highres_imu')     or {}
        ctrl = d.get('control_output')  or {}
        gate = d.get('gate_detection')
        nav  = d.get('navigation')      or {}
        vis  = d.get('vision')          or {}
        tel  = d.get('teleop_cmd')      or {}
        tgt  = d.get('planner_target')  or {}

        row = {
            # time
            't':              self._f(time.time() - self._t0),
            # attitude (from ATTITUDE msg)
            'roll':           self._f(att.get('roll')),
            'pitch':          self._f(att.get('pitch')),
            'yaw':            self._f(att.get('yaw')),
            'roll_rate':      self._f(att.get('rollspeed')),
            'pitch_rate':     self._f(att.get('pitchspeed')),
            'yaw_rate':       self._f(att.get('yawspeed')),
            # IMU (HIGHRES_IMU)
            'ax':             self._f(imu.get('xacc')),
            'ay':             self._f(imu.get('yacc')),
            'az':             self._f(imu.get('zacc')),
            'gx_imu':         self._f(imu.get('xgyro')),
            'gy_imu':         self._f(imu.get('ygyro')),
            'gz_imu':         self._f(imu.get('zgyro')),
            'baro':           self._f(imu.get('pressure_alt')),
            'temp':           self._f(imu.get('temperature'), 2),
            # control commands sent to sim
            'cmd_thrust':     self._f(ctrl.get('thrust')),
            'cmd_roll_rate':  self._f(ctrl.get('roll_rate')),
            'cmd_pitch_rate': self._f(ctrl.get('pitch_rate')),
            'cmd_yaw_rate':   self._f(ctrl.get('yaw_rate')),
            # planner target (NED velocity)
            'tgt_vn':         self._f(tgt.get('vn')),
            'tgt_ve':         self._f(tgt.get('ve')),
            'tgt_vd':         self._f(tgt.get('vd')),
            'tgt_yr':         self._f(tgt.get('yaw_rate')),
            # gate vision
            'gate_u':         self._f(gate['center_px'][0], 1) if gate else 'nan',
            'gate_v':         self._f(gate['center_px'][1], 1) if gate else 'nan',
            'gate_conf':      self._f(gate['confidence'],   3) if gate else 'nan',
            'gate_area':      self._f(gate['area_px'],      0) if gate else 'nan',
            'gate_method':    gate.get('method', 'none') if gate else 'none',
            'gate_predicted': str(int(bool(gate and gate.get('predicted')))),
            'gate_range':     self._f(vis.get('range_m')),
            # OpenCV body-frame navigator output
            'nav_state':      nav.get('state', 'none'),
            'nav_fwd':        self._f(nav.get('forward_mps')),
            'nav_right':      self._f(nav.get('right_mps')),
            'nav_down':       self._f(nav.get('down_mps')),
            'nav_yaw_rate':   self._f(nav.get('yaw_rate_rps')),
            'nav_align_err':  self._f(nav.get('alignment_error')),
            # planner mode
            'planner':        d.get('planner_mode', 'unknown'),
            # teleop inputs
            'tel_fwd':        str(int(tel.get('fwd',   0))),
            'tel_right':      str(int(tel.get('right', 0))),
            'tel_up':         str(int(tel.get('up',    0))),
            'tel_yaw':        str(int(tel.get('yaw',   0))),
        }

        if self._csv_writer is None:
            self._csv_file = open(self._csv_path, 'w', newline='')
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()

        self._csv_writer.writerow(row)
        self._csv_file.flush()
