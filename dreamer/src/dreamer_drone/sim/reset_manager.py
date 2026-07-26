"""Episode reset orchestration.

Sequence (see docs/system_architecture.md "Process lifecycle"):
  1. send vendor reset (cmd 31000)
  2. wait for the drone/race to settle (config.reset_settle_s)
  3. gate on stream health (fresh image + telemetry) before returning
  4. prime a short rate stream + arm so the ACRO controller is live

**(MEASURE)** cmd-31000 semantics: if it turns out to ACK or to reset active_gate->0, we
tighten step 3 to key on that instead of a fixed settle. Until then we use settle+health.
"""
from __future__ import annotations

import time

from ..config import SimConfig
from .action_sender import ActionSender
from .camera_io import CameraIO
from .mavlink_io import MavlinkIO
from .process_manager import SimUnavailable


class ResetManager:
    def __init__(self, io: MavlinkIO, cam: CameraIO, sender: ActionSender, cfg: SimConfig):
        self.io = io
        self.cam = cam
        self.sender = sender
        self.cfg = cfg

    def _streams_healthy(self) -> bool:
        # HIGHRES_IMU is the live telemetry stream in VQ2 (ATTITUDE is absent — measured).
        return (self.cam.age() < self.cfg.stale_image_s
                and self.io.age("highres_imu") < self.cfg.stale_telem_s)

    def wait_for_streams(self, timeout: float = 10.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._streams_healthy():
                return
            # keep the setpoint stream alive so the sim stays in control mode
            self.sender.send_neutral()
            time.sleep(0.02)
        raise SimUnavailable("streams did not become healthy after reset")

    def reset(self) -> None:
        self.sender.reset()
        self.io.send_reset()
        # prime a neutral setpoint stream during settle (ACRO stays live)
        t_end = time.time() + self.cfg.reset_settle_s
        while time.time() < t_end:
            self.sender.send_neutral()
            time.sleep(0.02)
        self.io.arm()
        self.wait_for_streams()
