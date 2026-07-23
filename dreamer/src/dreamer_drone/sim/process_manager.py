"""Launch / attach / kill the Windows DCL FlightSim from WSL.

The sim is a Windows GUI binary; from WSL we launch it via interop (running the `.exe`
path directly). Launching is *optional* — by default the env attaches to an
already-running sim (`SimConfig.launch_sim=False`), which is the safe path for a shared
desktop. Nothing here patches or copies the binary (compliance: installation preserved).
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Optional

from ..config import SimConfig


class SimUnavailable(RuntimeError):
    """Raised when the simulator cannot be reached or has died. The training loop
    catches this, relaunches, and resumes from the latest checkpoint."""


class ProcessManager:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.proc: Optional[subprocess.Popen] = None

    def launch(self) -> None:
        if not self.cfg.launch_sim:
            return  # attach mode: user starts the sim manually
        exe = self.cfg.sim_exe
        if not exe or not Path(exe).exists():
            raise SimUnavailable(f"sim_exe not found: {exe!r}")
        # WSL interop: executing a Windows .exe path launches it on the Windows host.
        self.proc = subprocess.Popen(
            [exe, *self.cfg.launch_args],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def is_alive(self) -> bool:
        if self.proc is None:
            return True  # attach mode: liveness is judged by stream watchdogs
        return self.proc.poll() is None

    def kill(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

    def relaunch(self) -> None:
        self.kill()
        time.sleep(2.0)
        self.launch()
