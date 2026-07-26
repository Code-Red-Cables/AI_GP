"""Typed, version-controlled configuration.

Every run loads a YAML config, and `Config.save()` writes the *fully resolved* config
back next to the run's artifacts so experiments are reproducible. Nothing in the code
base reads scattered module-level constants for tunables — they all live here.

NOTE: this module intentionally does NOT `from __future__ import annotations` — the YAML
loader introspects dataclass field types at runtime to rebuild nested sub-configs, which
requires the field `.type` to be the actual class object, not a string.
"""
import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #
@dataclass
class SimConfig:
    """Transport + process control for the DCL FlightSim SITL bridge."""
    mavlink_host: str = "127.0.0.1"     # set to the Windows host IP if not mirrored net
    mavlink_port: int = 14550
    camera_bind: str = "0.0.0.0"
    camera_port: int = 5600
    # process_manager: how to launch the Windows sim from WSL (optional; may be manual)
    sim_exe: str = ""                   # e.g. "/mnt/d/Code/Competitions/AI_GP/AIGP_3385/FlightSim.exe"
    launch_args: list[str] = field(default_factory=list)
    launch_sim: bool = False            # if False, attach to an already-running sim
    reset_cmd_id: int = 31000           # vendor MAVLINK_CMD_SIM_RESET
    heartbeat_timeout_s: float = 10.0
    reset_settle_s: float = 1.5         # wait after reset before first obs
    stale_image_s: float = 0.5          # watchdog: image stream considered dead after this
    stale_telem_s: float = 0.5          # watchdog: telemetry stream considered dead


@dataclass
class ObsConfig:
    image_h: int = 64
    image_w: int = 64
    grayscale: bool = False
    # vector obs is fixed-schema (see env/observation_builder.py); documented not configurable
    normalize_image_in_encoder: bool = True   # keep uint8 in replay, normalize in net


@dataclass
class ActionConfig:
    """Normalized policy output in [-1,1]^4 -> physical SET_ATTITUDE_TARGET."""
    hover_thrust: float = 0.27
    thrust_span: float = 0.5            # thrust = hover + a0 * span, then clamped
    thrust_min: float = 0.05
    thrust_max: float = 0.90
    # MEASURED (probe_control 2026-07-23): the sim applies ~2.5x gain to the rate command
    # (cmd 1.5 -> ~4 rad/s actual on roll/pitch, ~0.5x on yaw), so ±1 action * 3.0 spans a
    # sporty ~±8 rad/s roll/pitch. All three axes are inverted vs the gyro convention.
    max_rate_rad_s: float = 3.0         # |rate_i| = a_i * max_rate (command units, not literal rad/s)
    rate_sign_pitch: float = -1.0       # all axes inverted (measured)
    rate_sign_roll: float = -1.0
    rate_sign_yaw: float = -1.0
    slew_rate_limit: float = 0.0        # per-step max |Δa| in normalized units (0 = off)
    lpf_alpha: float = 0.0              # 0 = off; else a_out = (1-α)a_out + α a_in
    control_hz: float = 30.0            # decision rate; matches camera by default
    action_repeat: int = 1


@dataclass
class RewardConfig:
    w_progress: float = 1.0
    w_gate: float = 10.0
    w_finish: float = 100.0
    w_time: float = 0.05               # per second of sim time
    w_collision: float = 20.0
    w_collision_gate: float = 20.0     # gate-frame strike (COLLISION id 1001); default same
                                       # as w_collision, set lower to make near-miss threading
                                       # attempts cheaper than ground/obstacle crashes
    w_control: float = 0.01
    # MEASURED 2026-07-23: a per-step off-course penalty makes stable hovering (-39 over 60s)
    # score WORSE than an instant crash (-20), so the agent learned to crash. Zeroed — the
    # POSITIVE vision-progress reward already rewards keeping the gate in view/approaching.
    w_offcourse: float = 0.0           # gate not visible (vision proxy) — off (caused reward inversion)
    w_alive: float = 0.0               # optional per-step survival bonus; keep small (<time penalty
                                       # * span) to avoid a hover-farming exploit. Off by default.
    use_privileged_progress: bool = False   # auto-enabled if position telemetry present
    progress_clip: float = 2.0         # clamp per-step progress reward (anti-exploit)
    # Temporal consistency for the vision progress proxy (2026-07-24): the detector can
    # switch targets between frames (gate -> sign -> next gate), which produced +-0.9/step
    # reward spikes from comparing areas of DIFFERENT objects. If sqrt(area) or the
    # detection center jumps more than these thresholds in one step, treat it as a target
    # switch: re-baseline and emit 0 progress instead of a spurious delta.
    progress_area_jump: float = 1.35   # max allowed per-step sqrt(area) ratio
    progress_center_jump: float = 0.25 # max allowed per-step center move (frac of frame diag)
    # Potential-based centering shaping (2026-07-25): pays w_center * Δφ where
    # φ = -|u - 0.5| of the detected gate center (φ = -0.5 when no gate is visible).
    # Loops sum to zero (pure potential), so it cannot be farmed; what it DOES do is
    # charge the "slide sideways past the gate" trajectory (measured: episodes earned
    # +1.3-2.3 progress while the gate drifted out the left of frame, 0 passes) and
    # pay for re-centering / re-acquiring. Horizontal-only: vertical aim is
    # intentionally off-center (20 deg camera up-tilt).
    w_center: float = 0.0
    # Close-range hold: if the gate's sqrt(area) at last sight was >= this (full-res
    # detector px; 0 disables), losing it is treated as crossing the aperture and φ is
    # held neutral for up to center_hold_steps frames instead of charged. Without this
    # the centering term taxes the pass itself (the gate must exit the frame to fly
    # through it) and the policy learns to bail upward at the aperture.
    center_hold_sqrt_px: float = 0.0
    center_hold_steps: int = 8


@dataclass
class TerminationConfig:
    episode_timeout_s: float = 60.0    # sim-time budget per episode (< spec 8 min)
    collision_terminates: bool = True
    collision_threat_min: int = 1      # threat level that counts as terminal
    finish_terminates: bool = True
    stuck_no_gate_s: float = 12.0      # truncate if no gate progress for this long


@dataclass
class CurriculumConfig:
    enabled: bool = True
    # stage names map to reset distributions in env/curriculum.py
    stages: list[str] = field(default_factory=lambda: [
        "hover", "single_gate", "random_near_gate", "two_gate",
        "short_segment", "full_course", "full_course_fast", "recovery",
    ])
    start_stage: int = 0
    promote_success_rate: float = 0.7   # advance when rolling success >= this
    promote_window: int = 20            # over this many episodes


@dataclass
class ModelConfig:
    """DreamerV3 world model + actor-critic sizes."""
    # RSSM
    deter_dim: int = 512               # GRU deterministic state h
    stoch_dim: int = 32                # number of categorical variables z
    stoch_classes: int = 32            # classes per categorical
    hidden: int = 256                  # MLP width across the model
    # encoders
    cnn_depth: int = 32                # base channels of the image CNN
    vector_layers: int = 2
    # heads
    reward_bins: int = 255             # two-hot symlog reward/critic support
    critic_bins: int = 255
    free_bits: float = 1.0             # KL free bits (nats)
    kl_balance: float = 0.8


@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "cuda"               # falls back to cpu if unavailable
    batch_size: int = 16
    seq_len: int = 50
    replay_capacity: int = 500_000     # transitions
    prefill: int = 5_000               # random/scripted steps before learning
    train_ratio: int = 512             # gradient steps per env step ratio (Dreamer "train ratio")
    imag_horizon: int = 15
    gamma: float = 0.997
    lambda_: float = 0.95
    lr_model: float = 1e-4
    lr_actor: float = 3e-5
    lr_critic: float = 3e-5
    actor_entropy: float = 3e-4
    grad_clip: float = 1000.0
    total_env_steps: int = 2_000_000
    checkpoint_every: int = 10_000
    eval_every: int = 20_000
    log_dir: str = "artifacts/runs"
    # decoupled trainer
    max_episode_steps: int = 3_000     # hard cap per episode (termination usually fires first)
    sync_every: int = 100              # learner updates between collector weight syncs
    collector_device: str = "cpu"      # inference for collection (cheap at 30 Hz; frees GPU)
    log_every: int = 50                # learner updates between console/CSV log lines
    # behavior cloning from demos (--demos): warm-start the actor to imitate gate-passing runs,
    # then anneal to 0 so RL takes over. Fixes "world model learns but actor never passes a gate".
    bc_coef: float = 1.0               # initial BC weight on the actor loss
    bc_anneal: int = 30_000            # updates over which BC weight decays 1->0
    # MEASURED 2026-07-24: every collected demo ends in a collision shortly after gate 1,
    # so BC was cloning the crash approach too. Drop the last N steps of crash-ending
    # demos from the BC buffer only (the world-model replay keeps full episodes so it
    # still learns what collisions look like). ~30 steps ~= the final 1-2 s of flight.
    demo_trim_steps: int = 30
    # MEASURED 2026-07-25 (run 1784940039, 255k steps): gate-pass states were ~3% of
    # replay and shrinking, so imagination almost never started near an aperture and the
    # single live +40 pass could not anchor the value function. Mix this fraction of
    # every world-model batch from the demo buffer: imagination then regularly rolls out
    # from demo aperture states, where the gate reward is a step away.
    demo_wm_frac: float = 0.25


@dataclass
class Config:
    name: str = "dreamer_small"
    sim: SimConfig = field(default_factory=SimConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ---- (de)serialization -------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        kwargs: dict[str, Any] = {}
        fields = {f.name: f for f in dataclasses.fields(cls)}
        for key, val in raw.items():
            if key not in fields:
                raise KeyError(f"unknown config key: {key!r}")
            ftype = fields[key].type
            if dataclasses.is_dataclass(ftype) and isinstance(val, dict):
                kwargs[key] = ftype(**val)  # type: ignore[operator]
            else:
                kwargs[key] = val
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def load_config(path: str | Path | None) -> Config:
    """Load a config from YAML, or return defaults if path is None."""
    return Config.from_yaml(path) if path else Config()
