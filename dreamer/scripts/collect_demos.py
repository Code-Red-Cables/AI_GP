"""Collect replay episodes with a baseline policy and dump an inspectable dataset
(prompt Phase 4/5). Also the Milestone-2 environment smoke test.

    python dreamer/scripts/collect_demos.py --policy scripted --episodes 3
    # tune the stabilized baseline without editing code:
    python dreamer/scripts/collect_demos.py --policy scripted --forward-lean -0.12 --thrust-bias 0.05

Writes, per episode:
  artifacts/demos/episode_NNN.npz       replay arrays (image/vector/action/reward/cont)
  artifacts/demos/episode_NNN_log.csv   human-readable per-step log (commands + telemetry)
and prints a diagnostic summary (tumble check, thrust, gate-visibility, end reason).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from dreamer_drone.config import load_config  # noqa: E402
from dreamer_drone.env.ahrs import AHRSConfig  # noqa: E402
from dreamer_drone.env.baselines import (RandomPolicy, ScriptedController,  # noqa: E402
                                         StabilizedController)
from dreamer_drone.env.drone_racing_env import DroneRacingEnv  # noqa: E402
from dreamer_drone.env.spaces import VECTOR_OBS_FIELDS, scale_action  # noqa: E402
from dreamer_drone.dreamer.replay import EpisodeAccumulator  # noqa: E402

_FI = {n: i for i, n in enumerate(VECTOR_OBS_FIELDS)}
_LOG_HEADER = [
    "step", "thrust_cmd", "roll_cmd", "pitch_cmd", "yaw_cmd",
    "a_thrust", "a_roll", "a_pitch", "a_yaw",
    "gyro_x", "gyro_y", "gyro_z", "ax", "ay", "az", "tilt_roll", "tilt_pitch",
    "gate_visible", "reward", "r_collision", "r_progress", "r_gate",
    "active_gate", "term_reason",
]


def _log_row(step, action, phys, vec, info, reward):
    rc = info.get("reward_components", {})
    return [
        step, f"{phys.thrust:.3f}", f"{phys.roll_rate:+.3f}", f"{phys.pitch_rate:+.3f}",
        f"{phys.yaw_rate:+.3f}",
        f"{action[0]:+.3f}", f"{action[1]:+.3f}", f"{action[2]:+.3f}", f"{action[3]:+.3f}",
        f"{vec[_FI['gyro_x']]:+.3f}", f"{vec[_FI['gyro_y']]:+.3f}", f"{vec[_FI['gyro_z']]:+.3f}",
        f"{vec[_FI['ax']]:+.2f}", f"{vec[_FI['ay']]:+.2f}", f"{vec[_FI['az']]:+.2f}",
        f"{vec[_FI['tilt_roll']]:+.3f}", f"{vec[_FI['tilt_pitch']]:+.3f}",
        int(bool(info.get("gate_visible"))), f"{reward:+.3f}",
        f"{rc.get('collision', 0):+.2f}", f"{rc.get('progress', 0):+.3f}",
        f"{rc.get('gate_pass', 0):+.1f}",
        info.get("active_gate"), info.get("term_reason", ""),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--policy", choices=["random", "scripted", "stabilized"], default="stabilized")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--out", default="artifacts/demos")
    # baseline tuning (no code edits needed)
    ap.add_argument("--forward-lean", type=float, default=0.10)
    ap.add_argument("--thrust-bias", type=float, default=0.0, help="scripted: hover offset")
    ap.add_argument("--climb-bias", type=float, default=0.0, help="stabilized: thrust offset (0=hover)")
    ap.add_argument("--kp-att", type=float, default=0.6)
    ap.add_argument("--kd-att", type=float, default=0.03)
    ap.add_argument("--kp-yaw", type=float, default=0.8)
    ap.add_argument("--bank-gain", type=float, default=0.3)
    ap.add_argument("--kp-vert", type=float, default=0.4, help="gate vertical servo -> thrust (thread the hole)")
    ap.add_argument("--gate-v-target", type=float, default=0.58, help="target gate height in frame (frac; >0.5 for the up-tilt)")
    ap.add_argument("--ahrs-alpha", type=float, default=0.95)
    # AHRS gyro signs — flip to +/-1 if the estimate diverges (watch the printed 'ahrs_div')
    ap.add_argument("--gyro-sign-roll", type=float, default=1.0)
    ap.add_argument("--gyro-sign-pitch", type=float, default=1.0)
    ap.add_argument("--gyro-sign-yaw", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.policy == "random":
        policy = RandomPolicy()
    elif args.policy == "scripted":
        policy = ScriptedController(cfg.action, forward_lean=args.forward_lean,
                                    kp_att=2.5, kp_yaw=args.kp_yaw, thrust_bias=args.thrust_bias)
        print(f"[collect] scripted: forward_lean={args.forward_lean} thrust_bias={args.thrust_bias}",
              flush=True)
    else:  # stabilized (AHRS demonstrator)
        ahrs_cfg = AHRSConfig(alpha=args.ahrs_alpha, gyro_sign_roll=args.gyro_sign_roll,
                              gyro_sign_pitch=args.gyro_sign_pitch, gyro_sign_yaw=args.gyro_sign_yaw)
        policy = StabilizedController(cfg.action, forward_lean=args.forward_lean,
                                      kp_att=args.kp_att, kd_att=args.kd_att, kp_yaw=args.kp_yaw,
                                      bank_gain=args.bank_gain, kp_vert=args.kp_vert,
                                      gate_v_target=args.gate_v_target, climb_bias=args.climb_bias,
                                      ahrs_cfg=ahrs_cfg)
        print(f"[collect] stabilized(AHRS): forward_lean={args.forward_lean} climb_bias={args.climb_bias} "
              f"kp_att={args.kp_att} kp_yaw={args.kp_yaw} kp_vert={args.kp_vert} gate_v_target={args.gate_v_target} "
              f"gyro_signs=({args.gyro_sign_roll:+.0f},{args.gyro_sign_pitch:+.0f},{args.gyro_sign_yaw:+.0f})",
              flush=True)

    env = DroneRacingEnv(cfg)
    for ep in range(args.episodes):
        obs, info = env.reset()
        if hasattr(policy, "reset"):
            policy.reset()   # reset the AHRS integrator between episodes
        acc = EpisodeAccumulator()
        rows, ep_reward, max_gate, gate_seen = [], 0.0, 0, 0
        for step in range(args.max_steps):
            action = policy(obs)
            nobs, reward, term, trunc, info = env.step(action)
            phys = scale_action(action, cfg.action)
            rows.append(_log_row(step, action, phys, obs["vector"], info, reward))
            acc.add(obs["image"], obs["vector"], action, reward, 0.0 if term else 1.0)
            obs = nobs
            ep_reward += reward
            gate_seen += int(bool(info.get("gate_visible")))
            if info.get("active_gate") is not None:
                max_gate = max(max_gate, int(info["active_gate"]))
            if term or trunc:
                break

        data = acc.flush()
        if data is not None:
            np.savez_compressed(out / f"episode_{ep:03d}.npz", **data)
        with open(out / f"episode_{ep:03d}_log.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(_LOG_HEADER)
            w.writerows(rows)

        _summary(ep, rows, ep_reward, max_gate, gate_seen, info)
        if hasattr(policy, "ahrs"):
            print(f"           ahrs_div={policy.ahrs.divergence:.3f} rad "
                  f"(persistently >0.3 ⇒ flip a --gyro-sign-*)", flush=True)
    env.close()
    print(f"[collect] wrote {args.episodes} episodes + per-step logs to {out}", flush=True)
    return 0


def _summary(ep, rows, ep_reward, max_gate, gate_seen, info):
    n = len(rows)
    if n == 0:
        print(f"[collect] ep {ep}: EMPTY (no steps)")
        return
    arr = np.array([[float(r[i]) for i in (9, 10, 11)] for r in rows])  # gyro xyz
    tilt = np.array([[float(r[15]), float(r[16])] for r in rows])       # tilt roll,pitch
    thrust = float(rows[-1][1])
    mean_abs_gyro = np.abs(arr).mean(axis=0)
    tumble = "TUMBLING (sustained rotation)" if mean_abs_gyro.max() > 0.6 else "stable-ish"
    print(f"[collect] ep {ep}: len={n} reward={ep_reward:.1f} max_gate={max_gate} "
          f"reason={info.get('term_reason')}", flush=True)
    print(f"           mean|gyro| xyz={mean_abs_gyro.round(2)} -> {tumble}", flush=True)
    print(f"           tilt_pitch start={tilt[0,1]:+.2f} end={tilt[-1,1]:+.2f}  "
          f"tilt_roll start={tilt[0,0]:+.2f} end={tilt[-1,0]:+.2f}", flush=True)
    print(f"           thrust_cmd={thrust:.2f}  gate_visible {gate_seen}/{n} steps "
          f"({100*gate_seen/n:.0f}%)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
