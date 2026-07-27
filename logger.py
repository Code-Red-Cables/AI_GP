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
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)

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
        pos  = d.get('local_position_ned') or {}
        race = d.get('race_status')     or {}
        vis  = d.get('vision')          or {}
        tel  = d.get('teleop_cmd')      or {}
        tgt  = d.get('planner_target')  or {}
        gate_body = vis.get('gate_body') or (None, None, None)
        gate_normal = vis.get('normal_body') or (None, None, None)
        gate_ned = vis.get('gate_ned') or (None, None, None)

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
            'requested_yaw_rate': self._f(
                ctrl.get('requested_yaw_rate')
            ),
            'measured_yaw_rate': self._f(
                ctrl.get('measured_yaw_rate')
            ),
            'yaw_rate_feedback': self._f(
                ctrl.get('yaw_rate_feedback')
            ),
            'ahrs_roll':      self._f(ctrl.get('ahrs_roll')),
            'ahrs_pitch':     self._f(ctrl.get('ahrs_pitch')),
            'desired_roll':   self._f(ctrl.get('desired_roll')),
            'desired_pitch':  self._f(ctrl.get('desired_pitch')),
            'vertical_lift_fraction': self._f(
                ctrl.get('vertical_lift_fraction')
            ),
            'hover_thrust':   self._f(ctrl.get('hover_thrust')),
            'thrust_adjustment': self._f(ctrl.get('thrust_adjustment')),
            'vertical_command': self._f(ctrl.get('vertical_command')),
            'vertical_velocity': self._f(ctrl.get('vertical_velocity')),
            'control_safety': ctrl.get('safety_reason') or 'none',
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
            'gate_pose_method': vis.get('method', 'none'),
            'gate_body_x':    self._f(gate_body[0]),
            'gate_body_y':    self._f(gate_body[1]),
            'gate_body_z':    self._f(gate_body[2]),
            'gate_normal_x':  self._f(gate_normal[0]),
            'gate_normal_y':  self._f(gate_normal[1]),
            'gate_normal_z':  self._f(gate_normal[2]),
            'gate_ned_n':     self._f(gate_ned[0]),
            'gate_ned_e':     self._f(gate_ned[1]),
            'gate_ned_d':     self._f(gate_ned[2]),
            'gate_pnp_error': self._f(
                vis.get('pnp_reprojection_error')
            ),
            # OpenCV body-frame navigator output
            'nav_state':      nav.get('state', 'none'),
            'nav_fwd':        self._f(nav.get('forward_mps')),
            'nav_req_fwd':    self._f(nav.get('requested_forward_mps')),
            'nav_frame_limited': str(int(bool(nav.get('framing_limited')))),
            'nav_frame_edge': self._f(nav.get('framing_edge')),
            'nav_right':      self._f(nav.get('right_mps')),
            'nav_down':       self._f(nav.get('down_mps')),
            'nav_yaw_rate':   self._f(nav.get('yaw_rate_rps')),
            'nav_align_err':  self._f(nav.get('alignment_error')),
            'gate_vel_x':      self._f(nav.get('gate_velocity_x')),
            'gate_vel_y':      self._f(nav.get('gate_velocity_y')),
            'nav_lead_x':      self._f(nav.get('horizontal_lead_error')),
            # optional simulator telemetry and pass confirmation
            'vel_n':          self._f(pos.get('vx')),
            'vel_e':          self._f(pos.get('vy')),
            'vel_d':          self._f(pos.get('vz')),
            'active_gate':    race.get('active_gate', 'nan'),
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
