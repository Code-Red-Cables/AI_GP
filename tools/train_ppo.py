"""On-policy PPO from a categorical seed, with a KL leash to the teacher.

The seed already leaves the pad. Naive PPO overwrites that. This script
fine-tunes the same TCN, samples the 21-bin head, and adds KL(student ||
frozen seed) so the launch cannot freely drift.

Reward is vision-only: gate index, finish, and keypoint geometry.
Crash ends the episode with no extra cost. No odometry. A flat
visibility bonus is not used — that paid the pad to sit and stare.

    .\\winvenv\\Scripts\\python.exe tools\\train_ppo.py --init models\\policy_seed_18.pt

Leave the sim at **1x**. Start a race and stay on the pad facing gate 1.
0.2x is for human coaching; it starves PPO of episodes.
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INIT = 'models/policy_seed_18.pt'

# config.py reads these at import time. Must be set before any local import
# that pulls in config (policy_planner, setup, main).
os.environ.setdefault('FLIGHT_MODE', 'policy')
os.environ.setdefault('EKF_USE_PNP', '0')
os.environ.setdefault('VISION_DISPLAY', '0')
os.environ.setdefault('TAKEOFF_DURATION_S', '0')
os.environ.setdefault('AUTO_RESET_ON_CRASH', '0')
os.environ.setdefault('RESET_SIM_ON_START', '1')
os.environ.setdefault('POLICY_LOOP_HZ', '20')
os.environ.setdefault('POLICY_WEIGHTS', DEFAULT_INIT)

from policy_net import load_policy, save_policy  # noqa: E402
from ekf.commanded_accel import BodyVelocityIntegrator  # noqa: E402
from policy_planner import observation_from_shared  # noqa: E402
from race_obs import (  # noqa: E402
    LABEL_DIM,
    LABEL_NAMES,
    SNAP_VISUAL,
    apply_visual_snap,
    approach_potential,
    bin_centers,
    labels_to_bins,
    stack_history,
    visible_span,
    visual_target_changed,
)

def _live_obs(shared_data, *, with_context, with_velocity, integ, dt=0.02):
    vel = None
    if with_velocity and integ is not None:
        from policy_planner import _live_attitude, _num
        import config as _cfg
        ctrl = shared_data.get('control_output') or {}
        hover = _num(
            ctrl.get('hover_thrust'),
            float(getattr(_cfg, 'HOVER_THRUST', 0.255)),
        )
        thrust = _num(ctrl.get('thrust'), hover)
        roll, pitch = _live_attitude(shared_data)
        imu = shared_data.get('highres_imu') or {}
        omega = np.array([
            _num(imu.get('xgyro')),
            _num(imu.get('ygyro')),
            _num(imu.get('zgyro')),
        ], dtype=np.float64)
        v = integ.step(dt, thrust, roll, pitch, omega, hover_trim=hover)
        vel = [float(v[0]), float(v[1]), float(v[2])]
        shared_data['cmd_vel_body'] = vel
    return observation_from_shared(
        shared_data,
        with_context=with_context,
        with_velocity=with_velocity,
        vel_body=vel,
    )


SIM_IP = '127.0.0.1'
SIM_PORT = 14550

TIME_COST = 0.02
APPROACH_W = 1.5
LOST_LOCK = 0.25
GATE_REWARD = 4.0
NEXT_GATE_BONUS = 3.0
FINISH_REWARD = 12.0
REACQUIRE = 0.4
PAD_SPAN = 0.28
PUNCH_SPAN = 0.40


def _n_vis(obs) -> float:
    if obs is None or len(obs) < 24:
        return 0.0
    return float(sum(obs[16:24]))


def step_reward(
    *,
    prev_gate: int,
    gate: int,
    finished: bool,
    crashed: bool = False,
    dt: float,
    prev_obs=None,
    obs=None,
    new_target: bool = False,
    n_vis: float | None = None,
) -> float:
    """Score passing gates. Shaping only for *closing* on this opening.

    Last run: 32 / 1383 episodes reached gate 1, none reached 2. Symmetric
    span×align deltas were mostly pad jitter and punch-through flicker.

    * +GATE_REWARD per increment; extra NEXT_GATE_BONUS once past gate 1.
    * Finish +FINISH_REWARD.
    * Approach: only *positive* span growth past pad size, scaled by how
      centered the lock is. Shrink / YOLO wobble is ignored.
    * Lost lock is a fly-away only while the gate is still small. A close
      lock that disappears is a punch, not a miss.
    * Reacquire after gate 1: finding the next opening is the current wall.
    * Crash ends the episode with no extra cost.
    """
    _ = crashed
    reward = -TIME_COST * max(0.02, float(dt))
    passed = gate > prev_gate
    if passed:
        reward += GATE_REWARD * float(gate - prev_gate)
        if gate >= 2:
            reward += NEXT_GATE_BONUS
    if finished:
        reward += FINISH_REWARD

    now_vis = float(n_vis) if n_vis is not None else _n_vis(obs)
    if not new_target and prev_obs is not None and obs is not None:
        prev_s = visible_span(prev_obs)
        span = visible_span(obs)
        if (
            prev_s is not None and span is not None
            and span >= PAD_SPAN
        ):
            ds = span - prev_s
            if ds > 0.0:
                phi = approach_potential(obs)
                align = (
                    float(phi) / span
                    if phi is not None and span > 1e-6
                    else 0.5
                )
                reward += APPROACH_W * ds * max(0.25, align)
        if _n_vis(prev_obs) >= 4.0 and now_vis < 2.0 and not passed:
            if prev_s is None or prev_s < PUNCH_SPAN:
                reward -= LOST_LOCK
        if gate >= 1 and _n_vis(prev_obs) < 2.0 and now_vis >= 4.0:
            reward += REACQUIRE
    return float(reward)


def compute_gae(rewards, values, dones, *, gamma: float, lam: float, last_value: float):
    """GAE-λ advantages and returns. ``values`` is length T (not T+1)."""
    t_steps = len(rewards)
    adv = np.zeros(t_steps, dtype=np.float32)
    nxt = 0.0
    next_value = float(last_value)
    for i in range(t_steps - 1, -1, -1):
        nonterminal = 0.0 if dones[i] else 1.0
        delta = rewards[i] + gamma * next_value * nonterminal - values[i]
        nxt = delta + gamma * lam * nonterminal * nxt
        adv[i] = nxt
        next_value = values[i]
    ret = adv + np.asarray(values, dtype=np.float32)
    return adv, ret


class ValueNet(nn.Module):
    def __init__(self, n_in: int, history: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(n_in) * int(history), 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)


def _active_gate(shared_data) -> int:
    race = shared_data.get('race_status') or {}
    try:
        return int(race.get('active_gate') or 0)
    except (TypeError, ValueError):
        return 0


def _finished(shared_data) -> bool:
    race = shared_data.get('race_status') or {}
    try:
        fns = float(race.get('race_finish_ns') or 0)
    except (TypeError, ValueError):
        return False
    return math.isfinite(fns) and fns > 0 and _active_gate(shared_data) >= 17


def _floor_hit(shared_data, monitor, now: float) -> bool:
    """Pad/world slam, even if the drone never climbed.

    CrashMonitor ignores Environment hits until peak_climb >= 0.35 m, so a
    failed launch flops on the floor until the 60 s timeout. PPO should
    end that episode after the arming grace and reset.
    """
    if now < float(getattr(monitor, 'grace_until', 0.0)):
        return False
    col = shared_data.get('collision') or {}
    if col.get('id') != 1002:
        return False
    try:
        impulse = float(col.get('impulse') or 0.0)
        ts_ns = int(col.get('ts') or 0)
    except (TypeError, ValueError):
        return False
    age_s = (time.time_ns() - ts_ns) * 1e-9 if ts_ns else 999.0
    return impulse >= 0.15 and 0.0 <= age_s <= 0.5


class Rollout:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.logp = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, action, logp, reward, done, value):
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.int64))
        self.logp.append(float(logp))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))

    def __len__(self):
        return len(self.rewards)

    def clear(self):
        self.__init__()


def _apply_env(args) -> None:
    os.environ['FLIGHT_MODE'] = 'policy'
    os.environ['POLICY_WEIGHTS'] = str(args.init)
    os.environ['EKF_USE_PNP'] = '0'
    os.environ['VISION_DISPLAY'] = '0'
    os.environ['TAKEOFF_DURATION_S'] = '0'
    os.environ['AUTO_RESET_ON_CRASH'] = '0'
    os.environ['POLICY_LOOP_HZ'] = repr(float(args.hz))
    os.environ['RESET_SIM_ON_START'] = '1'
    os.environ['EARLY_START_HOLD_S'] = '0'


def _race_is_go(shared_data) -> bool:
    race = shared_data.get('race_status') or {}
    try:
        boot = int(race.get('sim_boot_ms'))
        start = int(race.get('race_start_ms'))
    except (TypeError, ValueError):
        return False
    return start >= 0 and boot >= start


def _reset_and_arm(
    controller,
    shared_data,
    state_estimator,
    history_buf,
    *,
    planner=None,
    with_context: bool = False,
    with_velocity: bool = False,
    integ=None,
    period: float = 0.05,
    history: int = 64,
) -> None:
    shared_data['flight_started'] = False
    shared_data['vision_reset_episode'] = True
    shared_data['gate_detection'] = None
    shared_data['collision'] = None
    shared_data['local_position_ned'] = None
    try:
        controller.disarm()
    except Exception:
        pass
    controller.send_sim_reset()
    import config
    time.sleep(max(0.3, min(0.5, float(getattr(config, 'SIM_RESET_SETTLE_S', 1.0)))))
    if state_estimator is not None:
        resetter = getattr(state_estimator, 'reset_episode', None)
        if callable(resetter):
            resetter()
    history_buf.clear()
    if planner is not None:
        buf = getattr(planner, '_buf', None)
        if buf is not None:
            buf.clear()
        plans = getattr(planner, '_plans', None)
        if plans is not None:
            plans.clear()
        reset_vel = getattr(planner, 'reset_episode', None)
        if callable(reset_vel):
            reset_vel()
    if integ is not None:
        integ.reset()

    # Countdown is when the sim unlocks. Use that time to fill H=64 the
    # same way the flyer does. A one-frame pad window argmaxes to idle.
    deadline = time.monotonic() + 8.0
    need = max(16, min(int(history), 64))
    while time.monotonic() < deadline:
        obs = _live_obs(
            shared_data,
            with_context=with_context,
            with_velocity=with_velocity,
            integ=integ,
            dt=period,
        )
        if SNAP_VISUAL and history_buf and visual_target_changed(history_buf[-1], obs):
            apply_visual_snap(history_buf, obs)
        history_buf.append(obs)
        if planner is not None:
            planner.compute_target(shared_data)
        if _race_is_go(shared_data) and len(history_buf) >= need:
            break
        time.sleep(max(0.02, float(period)))

    det = shared_data.get('gate_detection') or {}
    for _ in range(40):
        if det.get('center_px') is not None:
            break
        time.sleep(0.02)
        det = shared_data.get('gate_detection') or {}
    controller.arm()
    shared_data['flight_started'] = True
    shared_data['control_authority'] = 'policy'


def _hover(controller, config) -> None:
    shared_data = getattr(controller, 'data', None)
    if shared_data is None:
        return
    shared_data['planner_target'] = {
        'kalman': True,
        'acro': True,
        'unrestricted_rates': True,
        'roll_rate': 0.0,
        'pitch_rate': 0.0,
        'yaw_rate': 0.0,
        'thrust': float(config.HOVER_THRUST),
        'desired_roll': 0.0,
        'desired_pitch': 0.0,
    }
    for _ in range(6):
        controller.update()
        time.sleep(0.02)


def ppo_update(actor, teacher, critic, opt, rollout, *, args, device, last_value: float):
    if len(rollout) < 8:
        return {}
    obs = torch.as_tensor(np.stack(rollout.obs), device=device)
    actions = torch.as_tensor(np.stack(rollout.actions), device=device)
    old_logp = torch.as_tensor(rollout.logp, dtype=torch.float32, device=device)
    adv, ret = compute_gae(
        rollout.rewards, rollout.values, rollout.dones,
        gamma=args.gamma, lam=args.lam, last_value=last_value,
    )
    adv_t = torch.as_tensor(adv, device=device)
    ret_t = torch.as_tensor(ret, device=device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    n = len(rollout)
    idx = np.arange(n)
    info = {}
    for _ in range(args.ppo_epochs):
        np.random.shuffle(idx)
        for start in range(0, n, args.batch):
            mb = idx[start:start + args.batch]
            mb_t = torch.as_tensor(mb, device=device)
            logits = actor(obs[mb_t])[:, 0]          # (B, 4, bins)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(actions[mb_t]).sum(dim=-1)
            ratio = torch.exp(logp - old_logp[mb_t])
            clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
            policy_loss = -(torch.min(ratio * adv_t[mb_t], clipped * adv_t[mb_t])).mean()
            value_loss = 0.5 * (critic(obs[mb_t]) - ret_t[mb_t]).pow(2).mean()
            entropy = dist.entropy().sum(dim=-1).mean()
            with torch.no_grad():
                t_logits = teacher(obs[mb_t])[:, 0]
            t_prob = torch.softmax(t_logits, dim=-1)
            s_log = torch.log_softmax(logits, dim=-1)
            kl = (t_prob * (t_prob.clamp_min(1e-8).log() - s_log)).sum(dim=-1).mean()
            loss = (
                policy_loss
                + args.vf_coef * value_loss
                - args.entropy * entropy
                + args.kl_coef * kl
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), 0.5
            )
            opt.step()
            info = {
                'policy': float(policy_loss.detach()),
                'value': float(value_loss.detach()),
                'entropy': float(entropy.detach()),
                'kl': float(kl.detach()),
            }
    return info


def run(args) -> int:
    _apply_env(args)
    import config
    config.POLICY_WEIGHTS = str(args.init)
    from main import CrashMonitor, _wait_pad_vision
    from setup import setup_components

    init = Path(args.init)
    if not init.is_file():
        fallback = Path('models/policy_seed_17.pt')
        if fallback.is_file():
            print(f'[PPO] {init} missing — using {fallback}', flush=True)
            init = fallback
            os.environ['POLICY_WEIGHTS'] = str(init)
            config.POLICY_WEIGHTS = str(init)
        else:
            print(f'[PPO] no seed weights at {init}', flush=True)
            return 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actor, blob = load_policy(init, map_location=device)
    teacher, _ = load_policy(init, map_location=device)
    actor.to(device).eval()
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if int(actor.bins or 0) <= 0:
        print('[PPO] seed must be a categorical (bins) policy', flush=True)
        return 2
    history = int(actor.history)
    n_in = int(actor.n_in)
    bins = int(actor.bins)
    centers = np.stack(
        [np.asarray(bin_centers(bins)[n], dtype=np.float64) for n in LABEL_NAMES]
    )
    critic = ValueNet(n_in, history).to(device)
    opt = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=args.lr,
    )
    print(
        f'[PPO] teacher={init}  H={history} bins={bins} chunk={actor.chunk}  '
        f'device={device}  kl_coef={args.kl_coef}  lr={args.lr}',
        flush=True,
    )
    print('[PPO] sim at 1x. Start a race, pad facing gate 1.', flush=True)

    shared_data = {}
    components = setup_components(
        shared_data, int(time.time() * 1000), SIM_IP, SIM_PORT,
    )
    controller = components['controller']
    planner = components.get('planner')
    state_estimator = components.get('state_estimator')
    with_context = bool(blob.get('context', False))
    with_velocity = bool(blob.get('velocity', False)) or int(actor.n_in) in (32, 51)
    vel_integ = BodyVelocityIntegrator() if with_velocity else None
    monitor = CrashMonitor()
    history_buf: collections.deque = collections.deque(maxlen=history)
    period = 1.0 / max(1.0, float(args.hz))
    rollout = Rollout()
    updates = 0
    episodes = 0
    best_gate = 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        _wait_pad_vision(shared_data)
        _reset_and_arm(
            controller, shared_data, state_estimator, history_buf,
            planner=planner, with_context=with_context,
            with_velocity=with_velocity, integ=vel_integ,
            period=period, history=history,
        )
        monitor.note_armed(shared_data)
        prev_gate = _active_gate(shared_data)
        prev_obs = None
        ep_ret = 0.0
        ep_t0 = time.monotonic()

        while updates < args.updates:
            now = time.monotonic()
            obs_row = _live_obs(
                shared_data,
                with_context=with_context,
                with_velocity=with_velocity,
                integ=vel_integ,
                dt=period,
            )
            snapped = bool(
                SNAP_VISUAL
                and history_buf
                and visual_target_changed(history_buf[-1], obs_row)
            )
            if snapped:
                apply_visual_snap(history_buf, obs_row)
            history_buf.append(obs_row)
            window = stack_history(history_buf, history)
            x = torch.as_tensor([window], dtype=torch.float32, device=device)
            launching = (
                (time.monotonic() - ep_t0) < float(args.launch_s)
                and prev_gate <= 0
            )
            with torch.no_grad():
                logits = actor(x)[:, 0]
                dist = Categorical(logits=logits)
                value = float(critic(x)[0])
                if launching and planner is not None:
                    # Same decode the timed flyer uses (ensemble + bin window).
                    # Hard argmax of chunk-0 is the idle bin on the pad.
                    tgt = planner.compute_target(shared_data)
                    cmd = [
                        float(tgt['thrust']),
                        float(tgt['roll_rate']),
                        float(tgt['pitch_rate']),
                        float(tgt['yaw_rate']),
                    ]
                    idx = np.asarray(labels_to_bins(cmd, bins), dtype=np.int64)
                    action = torch.as_tensor([idx], device=device)
                else:
                    action = dist.sample()
                    idx = action[0].detach().cpu().numpy()
                    cmd = [float(centers[c, int(idx[c])]) for c in range(LABEL_DIM)]
                logp = float(dist.log_prob(action).sum())
            thrust = float(np.clip(cmd[0], config.MIN_THRUST, config.MAX_THRUST))
            if prev_obs is None:
                print(
                    f'[PPO] launch cmd  T={cmd[0]:+.3f}  R={cmd[1]:+.2f}  '
                    f'P={cmd[2]:+.2f}  Y={cmd[3]:+.2f}  flyer_launch={launching}',
                    flush=True,
                )
            shared_data['control_authority'] = 'policy'
            shared_data['planner_target'] = {
                'vn': 0.0, 've': 0.0, 'vd': 0.0,
                'kalman': True, 'acro': True, 'unrestricted_rates': True,
                'thrust': thrust,
                'roll_rate': cmd[1],
                'pitch_rate': cmd[2],
                'yaw_rate': cmd[3],
                'desired_roll': 0.0,
                'desired_pitch': 0.0,
            }
            controller.update()

            gate = _active_gate(shared_data)
            finished = _finished(shared_data)
            crashed = bool(monitor.update(shared_data, time.monotonic()))
            crashed = crashed or _floor_hit(shared_data, monitor, time.monotonic())
            timed_out = (time.monotonic() - ep_t0) >= args.episode_s
            done = finished or crashed or timed_out
            new_target = snapped or prev_obs is None or gate > prev_gate
            reward = step_reward(
                prev_gate=prev_gate, gate=gate,
                finished=finished, crashed=crashed,
                dt=period, prev_obs=prev_obs, obs=obs_row,
                new_target=new_target,
            )
            rollout.add(window, idx, logp, reward, done, value)
            ep_ret += reward
            prev_obs = list(obs_row)
            prev_gate = gate
            best_gate = max(best_gate, gate)

            remain = period - (time.monotonic() - now)
            if remain > 0:
                time.sleep(remain)

            if done:
                episodes += 1
                why = 'finish' if finished else ('crash' if crashed else 'timeout')
                span = visible_span(obs_row)
                print(
                    f'[PPO] ep={episodes}  {why}  gate={gate}  ret={ep_ret:.2f}  '
                    f'steps={len(rollout)}  best_gate={best_gate}  '
                    f'span={span if span is not None else float("nan"):.3f}',
                    flush=True,
                )
                if len(rollout) >= args.rollout:
                    _hover(controller, config)
                    last_v = 0.0
                    actor.train()
                    info = ppo_update(
                        actor, teacher, critic, opt, rollout,
                        args=args, device=device, last_value=last_v,
                    )
                    updates += 1
                    extra = {
                        'context': bool(blob.get('context', False)),
                        'chunk': int(actor.chunk),
                        'bins': int(actor.bins),
                        'ppo_update': updates,
                        'best_gate': best_gate,
                        'init': str(init),
                    }
                    save_policy(out, actor.cpu(), extra=extra)
                    actor.to(device)
                    actor.eval()
                    print(
                        f'[PPO] update={updates}  {info}  wrote {out}',
                        flush=True,
                    )
                    if updates % 10 == 0:
                        snap = out.with_name(f'{out.stem}_u{updates}{out.suffix}')
                        save_policy(snap, actor.cpu(), extra=extra)
                        actor.to(device)
                    rollout.clear()
                _reset_and_arm(
                    controller, shared_data, state_estimator, history_buf,
                    planner=planner, with_context=with_context,
                    with_velocity=with_velocity, integ=vel_integ,
                    period=period, history=history,
                )
                monitor.last_reset_at = time.monotonic()
                monitor.note_armed(shared_data)
                prev_gate = _active_gate(shared_data)
                prev_obs = None
                ep_ret = 0.0
                ep_t0 = time.monotonic()
    except KeyboardInterrupt:
        print('\n[PPO] interrupted', flush=True)
    finally:
        try:
            _hover(controller, config)
            controller.disarm()
        except Exception:
            pass
        logger = components.get('logger')
        if logger is not None:
            logger.stop()
        if len(rollout) >= 8:
            save_policy(out, actor.cpu(), extra={'context': bool(blob.get('context', False)),
                                                 'chunk': int(actor.chunk),
                                                 'bins': int(actor.bins),
                                                 'init': str(init)})
            print(f'[PPO] wrote {out}', flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--init', default=DEFAULT_INIT)
    ap.add_argument('--out', default='models/policy_ppo.pt')
    ap.add_argument('--hz', type=float, default=20.0)
    ap.add_argument('--updates', type=int, default=200)
    ap.add_argument('--rollout', type=int, default=1024)
    ap.add_argument('--episode-s', type=float, default=60.0)
    ap.add_argument(
        '--launch-s', type=float, default=2.5,
        help='seconds of teacher-argmax launch so the drone actually leaves '
             'the pad; then the student samples',
    )
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--clip', type=float, default=0.1)
    ap.add_argument('--kl-coef', type=float, default=0.15)
    ap.add_argument('--entropy', type=float, default=0.01)
    ap.add_argument('--vf-coef', type=float, default=0.5)
    ap.add_argument('--ppo-epochs', type=int, default=4)
    ap.add_argument('--batch', type=int, default=256)
    return ap


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
