"""Gymnasium-style training environment wrapping the DCL FlightSim SITL bridge.

    obs, info = env.reset(seed=None)
    obs, reward, terminated, truncated, info = env.step(action)   # action in [-1,1]^4

The env MAY read privileged state — but only for reward, termination, curriculum, and
info. The `obs` it returns is assembled by `observation_builder` from LEGAL fields only.
Real-time: `step()` transmits the action, waits ~one control period for the sim to
advance and a fresh camera frame to arrive, then reads the new observation.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config import Config
from ..sim.action_sender import ActionSender
from ..sim.camera_io import CameraIO
from ..sim.mavlink_io import MavlinkIO
from ..sim.privileged_state import PrivilegedState
from ..sim.process_manager import ProcessManager, SimUnavailable
from ..sim.reset_manager import ResetManager
from ..sim.timing import RateMeter
from .curriculum import Curriculum
from .observation_builder import build_obs
from .reward import RewardComputer, StepContext
from .spaces import ACTION_DIM, neutral_action
from .termination import TerminationChecker


def _load_gate_detector():
    """Reuse the proven repo-root detector if importable; else a no-op stub."""
    try:
        repo_root = Path(__file__).resolve().parents[4]  # dreamer/src/dreamer_drone/env -> repo
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from vision.gate_detector import detect_gate  # type: ignore
        return detect_gate
    except Exception:
        return None


class DroneRacingEnv:
    metadata = {"render_modes": []}

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.boot_ms = int(time.time() * 1000)
        self._detect_gate = _load_gate_detector()

        self.pm = ProcessManager(cfg.sim)
        self.io: Optional[MavlinkIO] = None
        self.cam: Optional[CameraIO] = None
        self.sender: Optional[ActionSender] = None
        self.priv: Optional[PrivilegedState] = None
        self.reset_mgr: Optional[ResetManager] = None

        self.reward_fn = RewardComputer(cfg.reward)
        self.term_fn = TerminationChecker(cfg.termination)
        self.curriculum = Curriculum(cfg.curriculum)

        self._prev_action = neutral_action()
        self._last_frame_id = -1
        self._last_sim_time_ns = 0
        self._prev_sim_time_s = 0.0
        self._ep_start_gate = 0
        self._ep_max_gate = 0
        self._ep_steps = 0
        self._cam_meter = RateMeter()
        self._connected = False

    # ---- connection --------------------------------------------------------
    def connect(self) -> None:
        if self._connected:
            return
        self.pm.launch()
        self.io = MavlinkIO(self.cfg.sim.mavlink_host, self.cfg.sim.mavlink_port,
                            self.boot_ms, self.cfg.sim.reset_cmd_id)
        if not self.io.wait_heartbeat(self.cfg.sim.heartbeat_timeout_s):
            raise SimUnavailable("no MAVLink heartbeat")
        self.io.start()
        self.cam = CameraIO(self.cfg.sim.camera_bind, self.cfg.sim.camera_port).start()
        self.sender = ActionSender(self.io, self.cfg.action)
        self.priv = PrivilegedState(self.io)
        self.reset_mgr = ResetManager(self.io, self.cam, self.sender, self.cfg.sim)
        self._connected = True

    def close(self) -> None:
        if self.sender:
            self.sender.send_neutral()
        if self.cam:
            self.cam.stop()
        if self.io:
            self.io.stop()
        self.pm.kill()
        self._connected = False

    # ---- gym API -----------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> tuple[dict, dict]:
        if not self._connected:
            self.connect()
        assert self.reset_mgr and self.priv and self.cam
        self.reset_mgr.reset()

        self._prev_action = neutral_action()
        self.sender.reset()  # type: ignore[union-attr]
        snap = self.priv.snapshot()
        self.priv.reset()
        self.reward_fn.reset()
        self.term_fn.reset(snap.active_gate)
        self._prev_sim_time_s = snap.sim_time_s
        self._ep_start_gate = int(snap.active_gate or 0)
        self._ep_max_gate = self._ep_start_gate
        self._ep_steps = 0

        frame = self.cam.get_latest()
        self._last_frame_id = frame.frame_id if frame else -1
        self._last_sim_time_ns = frame.sim_time_ns if frame else 0

        obs = self._build_obs(dt=0.0, image_valid=frame is not None)
        info = {"stage": self.curriculum.stage_name, "reset_reason": "reset"}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        assert self._connected and self.sender and self.priv and self.cam
        # transmit (possibly repeated) then wait for the next fresh camera frame
        applied = self._prev_action
        for _ in range(max(1, self.cfg.action.action_repeat)):
            applied = self.sender.send(action)
        image_valid, dt_cam = self._await_frame()

        snap = self.priv.snapshot()
        dt_sim = max(0.0, snap.sim_time_s - self._prev_sim_time_s)
        self._prev_sim_time_s = snap.sim_time_s

        # Post-reset collision grace: the previous episode's crash keeps emitting
        # collision events for a few frames after respawn, which killed (and -20
        # penalized) fresh episodes 2-10 steps in with a clean spawn view (measured
        # 2026-07-25, ~10% of episodes). The drone spawns hovering in free space, so
        # a REAL collision inside the first 8 steps is impossible.
        self._ep_steps += 1
        collision_threat = snap.collision_threat if self._ep_steps > 8 else 0

        # LEGAL vision signal for the proxy progress + off-course term
        gate_visible, gate_area, gate_center = self._vision()

        ctx = StepContext(
            dt_sim=dt_sim if dt_sim > 0 else dt_cam,
            active_gate=snap.active_gate, num_gates=snap.num_gates,
            finished=snap.finished, collision_threat=collision_threat,
            collision_is_gate=snap.collision_is_gate,
            dist_to_gate=snap.dist_to_gate,
            gate_area_px=gate_area, gate_visible=gate_visible, gate_center=gate_center,
            action=list(np.asarray(action, dtype=np.float32).reshape(-1)),
            prev_action=list(self._prev_action),
        )
        rc = self.reward_fn.compute(ctx)
        term = self.term_fn.check(ctx.dt_sim, snap.active_gate,
                                  snap.finished, collision_threat)

        # Late gate-pass grace (2026-07-26): race_status lags the physical crossing
        # (~0.75s update period measured), so a thread-then-clip crash can terminate
        # the episode BEFORE the sim reports the pass — the +40 (the whole point of
        # the episode) silently vanished. On a collision ending, poll briefly for a
        # late active_gate increment and credit it to this final step.
        if term.terminated and term.reason == "collision":
            base_gate = int(snap.active_gate) if snap.active_gate is not None else None
            deadline = time.time() + 1.2
            while time.time() < deadline:
                time.sleep(0.1)
                late = self.priv.snapshot()
                if (late.active_gate is not None and base_gate is not None
                        and int(late.active_gate) > base_gate):
                    delta = int(late.active_gate) - base_gate
                    rc.gate_pass += self.cfg.reward.w_gate * delta
                    rc.total += self.cfg.reward.w_gate * delta
                    rc.raw["late_gate_pass"] = float(delta)
                    snap = late   # info/_ep_max_gate below see the corrected gate
                    print(f"[env] late gate pass credited (+{delta} gate) after collision",
                          flush=True)
                    break

        if snap.active_gate is not None:
            self._ep_max_gate = max(self._ep_max_gate, int(snap.active_gate))

        self._prev_action = applied
        obs = self._build_obs(dt=dt_cam, image_valid=image_valid)
        gates_passed = self._ep_max_gate - self._ep_start_gate
        info = {
            "reward_components": rc.as_dict(),
            "active_gate": snap.active_gate,
            "gates_passed": gates_passed,
            "sim_time": snap.sim_time_s,
            "gate_visible": gate_visible,
            "term_reason": term.reason,
            "stage": self.curriculum.stage_name,
        }
        if term.terminated or term.truncated:
            success = self.curriculum.episode_success(term.reason, gates_passed)
            promoted = self.curriculum.record_episode(success)
            info["episode_success"] = success
            info["curriculum_promoted"] = promoted
        return obs, rc.total, term.terminated, term.truncated, info

    # ---- helpers -----------------------------------------------------------
    def _await_frame(self) -> tuple[bool, float]:
        """Block until a new camera frame arrives or the control period elapses.
        Returns (got_fresh_frame, dt_seconds since previous processed frame)."""
        period = 1.0 / max(1e-3, self.cfg.action.control_hz)
        deadline = time.time() + max(period, self.cfg.sim.stale_image_s)
        while time.time() < deadline:
            f = self.cam.get_latest()  # type: ignore[union-attr]
            if f and f.frame_id != self._last_frame_id:
                dt = (f.sim_time_ns - self._last_sim_time_ns) / 1e9
                self._last_frame_id = f.frame_id
                self._last_sim_time_ns = f.sim_time_ns
                self._cam_meter.tick()
                return True, dt if dt > 0 else period
            time.sleep(0.001)
        return False, period  # stale: held observation, marked invalid

    def _vision(self) -> tuple[bool, Optional[float], Optional[tuple]]:
        """Returns (gate_visible, area_px, center normalized to [0,1]^2)."""
        if self._detect_gate is None:
            return False, None, None
        f = self.cam.get_latest()  # type: ignore[union-attr]
        if f is None:
            return False, None, None
        det = self._detect_gate(f.image_bgr)
        if det is None:
            return False, None, None
        h, w = f.image_bgr.shape[:2]
        center = (float(det.center_px[0]) / max(1, w), float(det.center_px[1]) / max(1, h))
        return True, float(det.area_px), center

    def _build_obs(self, dt: float, image_valid: bool) -> dict:
        f = self.cam.get_latest()  # type: ignore[union-attr]
        imu = self.io.get("highres_imu")    # type: ignore[union-attr]
        # telemetry liveness keys on HIGHRES_IMU: ATTITUDE is absent in VQ2 (measured)
        telem_valid = self.io.age("highres_imu") < self.cfg.sim.stale_telem_s  # type: ignore[union-attr]
        return build_obs(
            frame_bgr=f.image_bgr if f else None,
            imu=imu,
            prev_action_norm=self._prev_action, dt=dt,
            cfg=self.cfg.obs, image_valid=image_valid, telem_valid=telem_valid,
        )

    # convenience for probes / baselines
    def camera_stats(self) -> Optional[Any]:
        return self._cam_meter.stats()
