"""HG-DAgger student planner: rolling H-frame observation → body rates.

``FLIGHT_MODE=policy``. Emits the same ``kalman=True`` rate contract as the
classical race planner so ``controller.py`` does not need a third path.
Commands are *requested* rates (pre ``RATE_SIGN_*``); training undoes the
logged signs in ``race_obs.labels_from_row`` so the round-trip matches.

``main.py`` must never import the gamepad path — interventions live only in
``tools/tune_flight.py coach``.
"""
from __future__ import annotations

import collections
import math
import os
from typing import Deque, Optional

import numpy as np

import config
from race_obs import (
    BIN_DECODE_WINDOW,
    DEFAULT_HISTORY,
    FEATURE_DIM,
    LABEL_NAMES,
    SNAP_VISUAL,
    apply_visual_snap,
    bin_centers,
    build_observation,
    decode_bin_probs,
    stack_history,
    visual_target_changed,
)


def _num(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _live_attitude(shared_data: dict) -> tuple[float, float]:
    """Roll/pitch matching ``race_obs.attitude_from_row``'s source order.

    Training reads att_raw_* first, so flight has to as well: feeding the
    EKF's drifting belief online to a policy trained on the sim's ATTITUDE is
    the train/deploy mismatch this module exists to prevent.
    """
    raw = shared_data.get('attitude_raw') or {}
    r, p = _num(raw.get('roll'), math.nan), _num(raw.get('pitch'), math.nan)
    if math.isfinite(r) and math.isfinite(p):
        return r, p
    ctrl = shared_data.get('control_output') or {}
    r = _num(ctrl.get('ahrs_roll'), math.nan)
    p = _num(ctrl.get('ahrs_pitch'), math.nan)
    if math.isfinite(r) and math.isfinite(p):
        return r, p
    att = shared_data.get('attitude') or {}
    return _num(att.get('roll')), _num(att.get('pitch'))


def observation_from_shared(
    shared_data: dict,
    *,
    sort_by_u: bool = False,
    with_context: bool = False,
) -> list[float]:
    """Build one observation row from the live shared_data dict."""
    gate = shared_data.get('gate_detection') or {}
    kps = gate.get('keypoints_px')
    confs = gate.get('keypoint_confidences')
    imu = shared_data.get('highres_imu') or {}
    roll, pitch = _live_attitude(shared_data)
    gate_index = None
    if with_context:
        race = shared_data.get('race_status') or {}
        gate_index = race.get('active_gate')
    return build_observation(
        kps,
        confs,
        roll=roll,
        pitch=pitch,
        gx=_num(imu.get('xgyro')),
        gy=_num(imu.get('ygyro')),
        gz=_num(imu.get('zgyro')),
        sort_by_u=sort_by_u,
        gate_index=gate_index,
    )


class PolicyPlanner:
    """Causal TCN student over a rolling observation history."""

    def __init__(self, weights_path: str | None = None):
        self.name = 'policy(hg-dagger)'
        path = weights_path or os.environ.get(
            'POLICY_WEIGHTS', 'models/policy_seed_17.pt'
        )
        self.weights_path = path
        self._sort_by_u = False
        self._history = DEFAULT_HISTORY
        self._buf: Deque[list[float]] = collections.deque(maxlen=DEFAULT_HISTORY)
        self._model = None
        self._device = 'cpu'
        self._load(path)

    def _load(self, path: str) -> None:
        try:
            import torch
            from policy_net import load_policy
        except ImportError as exc:
            raise RuntimeError(
                'torch is required for FLIGHT_MODE=policy'
            ) from exc

        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, blob = load_policy(path, map_location=self._device)
        arch = blob.get('arch') or {}
        self._history = int(arch.get('history', DEFAULT_HISTORY))
        self._chunk = int(arch.get('chunk', 1))
        self._bins = int(arch.get('bins', 0))
        self._centers = (
            np.stack([
                np.asarray(bin_centers(self._bins)[n], dtype=np.float64)
                for n in LABEL_NAMES
            ]) if self._bins else None
        )
        self._decode_window = int(
            getattr(config, 'BIN_DECODE_WINDOW', BIN_DECODE_WINDOW)
        )
        self._sort_by_u = bool(blob.get('sort_by_u', False))
        self._with_context = bool(blob.get('context', False))
        self._expect_dim = int(arch.get('n_in', FEATURE_DIM))
        self._buf = collections.deque(maxlen=self._history)
        # Overlapping chunk predictions for the current and future steps; the
        # planner averages whatever is available for "now" (ACT-style temporal
        # ensembling), which smooths the command stream without adding lag.
        # With a categorical head these are probability tables, not values.
        self._plans: collections.deque = collections.deque(
            maxlen=max(1, self._chunk)
        )
        self._model = model.to(self._device)
        self._model.eval()
        self._torch = torch
        print(
            f'[POLICY] loaded {path}  H={self._history}  chunk={self._chunk}  '
            f'bins={self._bins or "off (regression)"}  '
            f'context={self._with_context}  device={self._device}  '
            f'sort_by_u={self._sort_by_u}',
            flush=True,
        )

    def reset_episode(self) -> None:
        self._buf.clear()
        self._plans.clear()

    def compute_target(self, shared_data: dict) -> dict:
        shared_data['planner_mode'] = self.name
        obs = observation_from_shared(
            shared_data,
            sort_by_u=self._sort_by_u,
            with_context=self._with_context,
        )
        if len(obs) != self._expect_dim:
            raise RuntimeError(
                f'observation dim {len(obs)} != checkpoint n_in '
                f'{self._expect_dim} (context={self._with_context})'
            )
        # New visual lock / GATE_PASSED: paint the current corners + gate id
        # across the history window so the TCN does not keep flying the
        # previous target. IMU (roll/pitch/gyro) on those frames stays.
        snapped = False
        if (
            SNAP_VISUAL
            and self._buf
            and visual_target_changed(self._buf[-1], obs)
        ):
            apply_visual_snap(self._buf, obs)
            self._plans.clear()
            snapped = True
        self._buf.append(obs)
        window = stack_history(self._buf, self._history)
        x = self._torch.tensor(
            [window], dtype=self._torch.float32, device=self._device
        )
        with self._torch.no_grad():
            out = self._model(x).detach().cpu().numpy()

        if self._bins:
            # out is (1, chunk, LABEL_DIM, bins) of logits. Average the
            # *probabilities* of overlapping plans, then decode. Averaging the
            # decoded values instead would reintroduce exactly the
            # mode-averaging the categorical head exists to avoid.
            logits = out[0]
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs /= np.maximum(1e-12, probs.sum(axis=-1, keepdims=True))
            self._plans.append(probs)
            acc = None
            n = 0
            for age, plan in enumerate(reversed(self._plans)):
                if age >= len(plan):
                    continue
                acc = plan[age] if acc is None else acc + plan[age]
                n += 1
            avg = acc / max(1, n)                     # (LABEL_DIM, bins)
            y = np.asarray(
                decode_bin_probs(avg, self._bins, window=self._decode_window),
                dtype=np.float64,
            )
        elif self._chunk == 1:
            y = out.reshape(-1)
        else:
            # out is (1, chunk, n_out): a plan for now and the next chunk-1
            # steps. Age every stored plan by one step and average their
            # predictions for the current instant.
            self._plans.append(out.reshape(self._chunk, -1))
            acc = None
            n = 0
            for age, plan in enumerate(reversed(self._plans)):
                if age >= len(plan):
                    continue
                acc = plan[age] if acc is None else acc + plan[age]
                n += 1
            y = (acc / max(1, n)).reshape(-1)

        thrust, roll_rate, pitch_rate, yaw_rate = (
            float(y[0]), float(y[1]), float(y[2]), float(y[3])
        )
        # Keep collective in the plant's safe band; rates are clipped by
        # controller.py against YAW_RATE_MAX / MAX_RATE.
        thrust = float(np.clip(
            thrust,
            float(config.MIN_THRUST),
            float(config.MAX_THRUST),
        ))
        target = {
            'vn': 0.0,
            've': 0.0,
            'vd': 0.0,
            'kalman': True,
            # The demonstrations were flown in acro, where controller.py leaves
            # commanded body rates unclipped. Deploying through the clipped
            # path would silently reshape the very actions being imitated.
            'acro': True,
            'unrestricted_rates': True,
            'roll_rate': roll_rate,
            'pitch_rate': pitch_rate,
            'yaw_rate': yaw_rate,
            'thrust': thrust,
            'desired_roll': 0.0,
            'desired_pitch': 0.0,
        }
        shared_data['planner_target'] = target
        shared_data['policy_obs'] = {
            'n_vis': float(sum(obs[16:24])),
            'history': len(self._buf),
            'visual_snap': snapped,
        }
        return target
