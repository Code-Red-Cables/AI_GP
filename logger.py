import csv
import math
import os
import threading
import time
from datetime import datetime

LOG_DIR = 'logs'
LOG_HZ  = float(os.environ.get('LOG_HZ', '50') or 50)

_NAN = float('nan')


class Logger:
    """
    Background thread that writes a CSV telemetry row at LOG_HZ and an
    events text file on demand.  Call log_event() from any thread.
    shared_data['log_event'] is set to this method so other components
    can log without importing this module.

    ``shared_data['log_hz']`` overrides the rate at runtime (used when the
    client is in slow-mo so H-frame history still spans similar sim time).
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
        shared_data.setdefault('log_hz', LOG_HZ)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f'[LOG] telem  -> {self._csv_path}', flush=True)
        print(f'[LOG] events -> {self._evt_path}', flush=True)

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
        while self._running:
            try:
                hz = float(self.data.get('log_hz') or LOG_HZ)
            except (TypeError, ValueError):
                hz = LOG_HZ
            hz = max(1.0, min(200.0, hz))
            interval = 1.0 / hz
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

    def _cand_best_conf(self, cands: dict) -> str:
        """Confidence of the strongest raw candidate box, or nan."""
        best = None
        for item in (cands.get('items') or ()):
            try:
                c = float(item.get('confidence'))
            except (TypeError, ValueError):
                continue
            if c == c and (best is None or c > best):
                best = c
        return self._f(best, 3)

    def _write_row(self):
        d    = self.data
        att  = d.get('attitude')        or {}
        # The sim's own ATTITUDE message. Spec section 4.3 lists it as
        # supported telemetry, and it is gravity-referenced, so unlike
        # shared_data['attitude'] (EKF-owned integrated gyro, measured drifting
        # +4 deg to -23 deg over 50 s) it is safe as a policy input. Logged
        # separately so a training run can never silently learn from the drift.
        att_raw = d.get('attitude_raw') or {}
        imu  = d.get('highres_imu')     or {}
        ctrl = d.get('control_output')  or {}
        gate = d.get('gate_detection')
        # Every box YOLO produced this frame, before identity selection and the
        # predicted/found gates in vision_rx dropped any of them. Without this a
        # frame where the detector saw the gate and the selection logic threw it
        # away is indistinguishable from a frame where nothing was in view.
        cands = d.get('gate_candidates') or {}
        snake = d.get('snake_gate')     or {}
        gatenet = d.get('gatenet')      or {}
        nav  = d.get('navigation')      or {}
        # VIO-owned position_ned when present; VQ1 sim odometry otherwise.
        pos  = d.get('position_ned') or d.get('local_position_ned') or {}
        # Raw sim ODOMETRY, kept strictly separate from the estimator's belief
        # above. pos_*/vel_* are dead reckoning and have been observed ranging
        # to 1e7 m; odo_* is measurement. Never conflate them when scoring.
        odo  = d.get('odometry')        or {}
        odo_q = odo.get('q') or (None, None, None, None)
        vio  = d.get('vio_stats')        or {}
        race = d.get('race_status')     or {}
        vis  = d.get('vision')          or {}
        dual = d.get('dual_gate_pnp')   or {}
        vision_nav = d.get('navigation') or {}
        tel  = d.get('teleop_cmd')      or {}
        race_pose = d.get('race_pose')  or {}
        tgt  = d.get('planner_target')  or {}
        ekf  = d.get('ekf_state')       or {}
        gyro_bias = ekf.get('gyro_bias') or (None, None, None)
        gate_body = (
            vis.get('gate_body') or dual.get('gate1_body')
            or (None, None, None)
        )
        gate_normal = (
            vis.get('normal_body') or dual.get('gate1_normal_body')
            or (None, None, None)
        )
        gate_ned = vis.get('gate_ned') or (None, None, None)
        gate_range = vis.get('range_m')
        if gate_range is None:
            gate_range = dual.get('gate1_range_m')
        gate_reproj = vis.get('pnp_reprojection_error')
        if gate_reproj is None:
            gate_reproj = dual.get('gate1_reproj_px')
        # Raw eight keypoints — the learned policy's actual input. Logged
        # per-corner rather than as the PnP-derived centre/range so a training
        # run does not silently depend on a PnP solve that usually fails.
        kps = (gate or {}).get('keypoints_px') or dual.get('keypoints_px')
        kconf = (
            (gate or {}).get('keypoint_confidences')
            or dual.get('keypoint_confidences')
        )
        kp_cols: dict[str, str] = {}
        for i in range(8):
            u = v = c = None
            if kps is not None and i < len(kps):
                try:
                    u, v = kps[i][0], kps[i][1]
                except (TypeError, IndexError):
                    u = v = None
            if kconf is not None and i < len(kconf):
                try:
                    c = kconf[i]
                except (TypeError, IndexError):
                    c = None
            kp_cols[f'kp{i}_u'] = self._f(u, 1)
            kp_cols[f'kp{i}_v'] = self._f(v, 1)
            kp_cols[f'kp{i}_c'] = self._f(c, 3)

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
            'gate_frame_id':  str(gate.get('frame_id', 'nan')) if gate else 'nan',
            'gate_ts_ns':     str(gate.get('ts', 'nan')) if gate else 'nan',
            'gate_cand_n':    str(len(cands.get('items') or ())),
            'gate_cand_conf': self._cand_best_conf(cands),
            'gate_cand_frame': str(cands.get('frame_id', 'nan')),
            'gate_reject':    str(cands.get('reject', '') or ''),
            'gate_raw_method': str(cands.get('method', '') or ''),
            # Observe-only snake gate detection, for comparison against YOLO.
            'snake_n':        str(int(snake.get('n', 0) or 0)),
            'snake_cf':       self._f(snake.get('best_fitness'), 3),
            'snake_ms':       self._f(snake.get('elapsed_ms'), 2),
            'snake_mask':     self._f(snake.get('mask_fraction'), 4),
            'gatenet_n':      str(int(gatenet.get('n_seen', 0) or 0)),
            'gatenet_score':  self._f(gatenet.get('min_score'), 3),
            'gatenet_ms':     self._f(gatenet.get('elapsed_ms'), 2),
            **kp_cols,
            'gate_norm_x':    self._f(dual.get('gate1_norm_x')),
            'gate_norm_y':    self._f(dual.get('gate1_norm_y')),
            'gate_range':     self._f(gate_range),
            'gate_pose_method': (
                vis.get('method', 'dual_pnp' if dual else 'none')
            ),
            'gate_pnp_held':  str(int(bool(dual.get('held')))),
            'gate_pnp_n':     str(int(dual.get('n_solved') or 0)),
            'vision_sim_time_ns': str(
                vision_nav.get('sim_time_ns', 'nan')
            ),
            'gate_body_x':    self._f(gate_body[0]),
            'gate_body_y':    self._f(gate_body[1]),
            'gate_body_z':    self._f(gate_body[2]),
            'gate_normal_x':  self._f(gate_normal[0]),
            'gate_normal_y':  self._f(gate_normal[1]),
            'gate_normal_z':  self._f(gate_normal[2]),
            'gate_ned_n':     self._f(gate_ned[0]),
            'gate_ned_e':     self._f(gate_ned[1]),
            'gate_ned_d':     self._f(gate_ned[2]),
            'gate_pnp_error': self._f(gate_reproj),
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
            # position/velocity belief (VIO when enabled, else sim odometry)
            'pos_n':          self._f(pos.get('x')),
            'pos_e':          self._f(pos.get('y')),
            'pos_d':          self._f(pos.get('z')),
            'vel_n':          self._f(pos.get('vx')),
            'vel_e':          self._f(pos.get('vy')),
            'vel_d':          self._f(pos.get('vz')),
            'att_source':     att.get('source', 'sim'),
            # sim ATTITUDE, unfiltered — the policy's attitude channel
            'att_raw_roll':   self._f(att_raw.get('roll')),
            'att_raw_pitch':  self._f(att_raw.get('pitch')),
            'att_raw_yaw':    self._f(att_raw.get('yaw')),
            'att_raw_rollspeed':  self._f(att_raw.get('rollspeed')),
            'att_raw_pitchspeed': self._f(att_raw.get('pitchspeed')),
            'att_raw_yawspeed':   self._f(att_raw.get('yawspeed')),
            # ground truth from the sim's ODOMETRY (VQ1 only; nan on VQ2)
            'odo_x':          self._f(odo.get('x')),
            'odo_y':          self._f(odo.get('y')),
            'odo_z':          self._f(odo.get('z')),
            'odo_vx':         self._f(odo.get('vx')),
            'odo_vy':         self._f(odo.get('vy')),
            'odo_vz':         self._f(odo.get('vz')),
            'odo_qw':         self._f(odo_q[0]),
            'odo_qx':         self._f(odo_q[1]),
            'odo_qy':         self._f(odo_q[2]),
            'odo_qz':         self._f(odo_q[3]),
            'odo_roll':       self._f(odo.get('roll')),
            'odo_pitch':      self._f(odo.get('pitch')),
            'odo_yaw':        self._f(odo.get('yaw')),
            'odo_rollspeed':  self._f(odo.get('rollspeed')),
            'odo_pitchspeed': self._f(odo.get('pitchspeed')),
            'odo_yawspeed':   self._f(odo.get('yawspeed')),
            # who is flying — HG-DAgger label provenance
            'control_authority': str(d.get('control_authority', 'policy')),
            'intervention_id':   str(d.get('intervention_id', '')),
            # Attempt index within this log. A sim reset teleports the drone,
            # so a training window must never span two attempts.
            'attempt':           str(d.get('attempt', 0)),
            # Operator-marked "do not learn from this" — repositioning after a
            # failure, ferrying to a section, anything not worth imitating.
            'exclude':           str(int(bool(d.get('exclude', 0)))),
            'vio_fixes':      str(vio.get('fixes', 0)),
            'vio_fix_rejects': str(vio.get('fixes_rejected', 0)),
            # vision attitude aid (gate horizon / yaw anchor)
            'gh_fixes':       str(ekf.get('gate_horizon_fixes', 0)),
            'gh_rejects':     str(ekf.get('gate_att_rejects', 0)),
            'gh_skips':       str(ekf.get('gate_att_skips', 0)),
            'gy_fixes':       str(ekf.get('gate_yaw_fixes', 0)),
            'gh_innov_r':     self._f(ekf.get('horizon_innov_roll')),
            'gh_innov_p':     self._f(ekf.get('horizon_innov_pitch')),
            'gy_innov':       self._f(ekf.get('yaw_innov')),
            'bias_gx':        self._f(gyro_bias[0]),
            'bias_gy':        self._f(gyro_bias[1]),
            'bias_gz':        self._f(gyro_bias[2]),
            'active_gate':    race.get('active_gate', 'nan'),
            'sim_boot_ms':    race.get('sim_boot_ms', 'nan'),
            'race_start_ms':  race.get('race_start_ms', 'nan'),
            'race_finish_ns': race.get('race_finish_ns', 'nan'),
            'last_gate_time_ns': race.get('last_gate_time', 'nan'),
            'race_rx_perf_s': self._f(
                race.get('received_perf_counter_s'), 9
            ),
            'race_rx_wall_ns': race.get('received_wall_time_ns', 'nan'),
            # planner mode
            'planner':        d.get('planner_mode', 'unknown'),
            # Classical race planner (FLIGHT_MODE=race): the LS gate solve it
            # steers on. Without these there is no way to tell "no gate in
            # view" from "gate seen but the solve was rejected".
            'race_mode':      str(race_pose.get('mode', 'none')),
            'race_range':     self._f(race_pose.get('range_m')),
            'race_lat':       self._f(race_pose.get('lateral_m')),
            'race_vert':      self._f(race_pose.get('vertical_m')),
            'race_bearing':   self._f(race_pose.get('bearing_rad')),
            'race_resid':     self._f(race_pose.get('residual_m')),
            'race_ring_dis':  self._f(race_pose.get('ring_disagree_m')),
            # assist / kalman path snapshot
            'path_phase':     str((d.get('kalman_path') or {}).get('phase', 'none')),
            'path_source':    str((d.get('kalman_path') or {}).get('source', 'none')),
            'path_nx':        self._f((d.get('kalman_path') or {}).get('norm_x')),
            'path_ny':        self._f((d.get('kalman_path') or {}).get('norm_y')),
            'path_thrust':    self._f((d.get('kalman_path') or {}).get('thrust')),
            'path_climbed':   self._f((d.get('kalman_path') or {}).get('climbed')),
            'path_vert_src':  str(
                (d.get('kalman_path') or {}).get('vert_src', 'none')
            ),
            'tgt_thrust':     self._f(tgt.get('thrust')),
            'tgt_roll_rate':  self._f(tgt.get('roll_rate')),
            'tgt_pitch_rate': self._f(tgt.get('pitch_rate')),
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
