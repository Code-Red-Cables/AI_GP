"""Train the racing policy from telemetry, with HG-DAgger aggregation.

Builds H-frame observation windows from telemetry CSVs and regresses the logged
command. Serves both roles in the plan:

  Phase 5  seed policy: plain behavior cloning on clean human laps. Expected to
           fly poorly -- paper1 Table 1 measures offline BC at 0% success on all
           three tracks. Its job is to produce failures worth correcting.
  Phase 6  DAgger rounds: pass every round's log, including the seed. Data is
           aggregated, never replaced (D = D_seed u D_round1 u ...), and
           intervention segments are weighted up.

Three details of the intervention handling matter, all from the HG-DAgger
literature:

  * Aggregate rather than replace. Training only on interventions makes the
    policy forget the nominal behaviour it already had.
  * Back-date each intervention by --lead-s. The moment the human grabs the
    sticks is already too late; the states leading in are where the correction
    should have begun, and those frames carry the human's label.
  * Keep a --tail-s recovery window after control returns, so the policy learns
    how a recovery ends rather than only how it starts.

    python tools/train_policy.py --telem logs/telem_A.csv logs/telem_B.csv
    python tools/train_policy.py --glob 'logs/telem_2026*.csv' --epochs 60
    python tools/train_policy.py --glob 'logs/seed/telem_*.csv' --glob 'logs/coach/telem_*.csv'
"""
from __future__ import annotations

import argparse
import csv
import glob as globlib
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from race_obs import (  # noqa: E402
    ACTION_RANGES,
    DEFAULT_HISTORY,
    FEATURE_DIM,
    FRAME_H,
    FRAME_W,
    KEYPOINT_COUNT,
    NOT_SEEN,
    SNAP_VISUAL,
    apply_visual_snap,
    bin_centers,
    feature_dim,
    LABEL_DIM,
    LABEL_NAMES,
    attitude_is_trusted,
    augment_corners,
    labels_from_row,
    observation_from_row,
    commanded_velocity_from_rows,
    visual_target_changed,
)


def _num(value, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


_COLLISION_RE = re.compile(
    r'\s*([\d.]+)\s+COLLISION\s+(\S+)\s+threat=(\d+)\s+impulse=([\d.]+)'
)

# Below this share of policy-flown frames a run counts as a pure human
# demonstration rather than a DAgger round with interventions.
DEMO_POLICY_FRAC = 0.02


def sim_time_scale(rows, times) -> float:
    """Sim seconds per wall second, from ``last_gate_time_ns`` vs ``t``.

    Seed logs at CE 0.2x stamp ``t`` with wall clock (~0.1 s/row) while
    physics advances 0.02 s/row. Striding on wall ``t`` then flying at 10 Hz
    holds the opening pitch-down for 5x too long and never reaches the
    look-up. 1.0 means the log is already in sim time (1x).
    """
    # last_gate_time_ns is the race clock at the last GATE_PASSED, held
    # constant until the next pass. Scale is (Δ race s) / (Δ wall s) across
    # those plateaus, not consecutive rows (a 1.3 s gate split is one row).
    plateaus: list[tuple[float, float]] = []
    prev_lg = None
    for row, t in zip(rows, times):
        ns = _num(row.get('last_gate_time_ns'))
        if not math.isfinite(t) or not math.isfinite(ns) or ns <= 0.0:
            continue
        lg = ns * 1e-9
        if prev_lg is None or lg > prev_lg + 1e-4:
            plateaus.append((t, lg))
            prev_lg = lg
    samples = []
    for (t0, l0), (t1, l1) in zip(plateaus, plateaus[1:]):
        if t1 > t0 + 0.2 and l1 > l0:
            samples.append((l1 - l0) / (t1 - t0))
    if len(samples) < 3:
        return 1.0
    scale = float(np.median(samples))
    if scale < 0.05 or scale > 1.01:
        return 1.0
    return scale


def collision_times(telem_path: Path, min_impulse: float = 0.0) -> list[float]:
    """Contact timestamps for a run, read from its sibling events file.

    The logger does not carry collisions in the CSV, but ``events_<ts>.txt``
    sits next to every ``telem_<ts>.csv`` and records each one. Parsing it here
    means existing runs can be filtered retroactively.
    """
    events = telem_path.with_name(
        telem_path.name.replace('telem_', 'events_')
    ).with_suffix('.txt')
    if not events.is_file():
        return []
    out: list[float] = []
    for line in events.read_text(errors='replace').splitlines():
        if 'COLLISION' not in line:
            continue
        m = _COLLISION_RE.match(line)
        if not m:
            continue
        try:
            t = float(m.group(1))
            impulse = float(m.group(4))
        except ValueError:
            continue
        if impulse >= min_impulse:
            out.append(t)
    return out


def load_run(
    path: Path,
    *,
    lead_s: float,
    tail_s: float,
    sort_by_u: bool,
    drop_collision_s: float = 0.0,
    min_impulse: float = 0.0,
    drop_policy_frames: bool = True,
    with_context: bool = False,
    with_velocity: bool = False,
):
    """Return (obs, labels, weight_hint, valid, marked, gates) for one run.

    ``weight_hint`` is 1.0 for policy-flown frames and 2.0 for frames inside an
    intervention window (including the back-dated lead and recovery tail); the
    caller scales it. ``valid`` marks rows whose label is finite, which have at
    least one visible corner, and whose attitude comes from a trusted source.

    ``drop_collision_s`` additionally invalidates rows within that many seconds
    of a logged contact. A behaviour-cloning seed should imitate clean flying,
    and a scrape along a gate is the one thing you never want copied.
    """
    with path.open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None

    times = [_num(r.get('t')) for r in rows]
    authority = [str(r.get('control_authority', 'policy')) for r in rows]
    human = np.array([a == 'human' for a in authority], dtype=bool)

    # A seed lap is human end to end, so there is no *intervention* in it --
    # marking every frame would weight the whole run up and mean nothing. Only
    # runs the policy actually flew have interventions. The threshold is a
    # fraction rather than "any policy row" because a couple of rows are logged
    # with the default authority before arm.
    is_dagger_round = float(np.mean(~human)) >= DEMO_POLICY_FRAC
    if not is_dagger_round:
        human = np.zeros(len(rows), dtype=bool)

    # Back-date and extend each contiguous human block.
    marked = human.copy()
    n = len(rows)
    i = 0
    while i < n:
        if not human[i]:
            i += 1
            continue
        start = i
        while i < n and human[i]:
            i += 1
        end = i - 1
        t_start = times[start]
        t_end = times[end]
        if math.isfinite(t_start):
            j = start
            while j >= 0 and math.isfinite(times[j]) and t_start - times[j] <= lead_s:
                marked[j] = True
                j -= 1
        if math.isfinite(t_end):
            j = end
            while j < n and math.isfinite(times[j]) and times[j] - t_end <= tail_s:
                marked[j] = True
                j += 1

    hits = collision_times(path, min_impulse) if drop_collision_s > 0.0 else []

    def _near_collision(t: float) -> bool:
        if not hits or not math.isfinite(t):
            return False
        return any(abs(t - h) <= drop_collision_s for h in hits)

    obs = np.zeros(
        (n, feature_dim(with_context, with_velocity)), dtype=np.float32
    )
    lab = np.zeros((n, LABEL_DIM), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    gates = np.full(n, -1, dtype=np.int32)
    attempts = np.zeros(n, dtype=np.int32)
    vel = (
        commanded_velocity_from_rows(rows)
        if with_velocity
        else None
    )
    for k, row in enumerate(rows):
        vx = vy = vz = 0.0
        if vel is not None:
            vx, vy, vz = vel[k]
        vec = observation_from_row(
            row,
            sort_by_u=sort_by_u,
            with_context=with_context,
            with_velocity=with_velocity,
            vx=vx,
            vy=vy,
            vz=vz,
        )
        obs[k] = vec
        y = labels_from_row(row)
        lab[k] = y
        # A row with no visible corner teaches the policy to fly blind from a
        # sentinel-only observation; the history window still covers dropouts.
        # An untrusted attitude row would feed the EKF's drift into the
        # attitude channel, which is the failure race_obs documents.
        n_vis = float(np.sum(obs[k][16:24]))
        valid[k] = (
            all(math.isfinite(v) for v in y)
            and n_vis > 0.0
            and attitude_is_trusted(row)
            and not _near_collision(times[k])
            # Operator marked this stretch as not worth imitating.
            and _num(row.get('exclude'), 0.0) < 0.5
        )
        g = _num(row.get('active_gate'))
        gates[k] = int(g) if math.isfinite(g) else -1
        a = _num(row.get('attempt'))
        attempts[k] = int(a) if math.isfinite(a) else 0

    if is_dagger_round and drop_policy_frames:
        # HG-DAgger trains on states the policy visited paired with the
        # *expert's* action -- which only exists where the human intervened.
        # On a policy-flown frame the recorded command IS the policy's own
        # output, so training on it is self-imitation: it teaches the model to
        # reproduce whatever it already does, including the failure that
        # prompted the round. Measured on one 137-minute session, keeping them
        # would have added 372k such frames against 39k real ones.
        valid &= marked

    weight = np.where(marked, 2.0, 1.0).astype(np.float32)
    steps = [b - a for a, b in zip(times, times[1:])
             if math.isfinite(a) and math.isfinite(b) and 0.0 < b - a < 1.0]
    wall_dt = float(np.median(steps)) if steps else 0.1
    scale = sim_time_scale(rows, times)
    row_dt = float(wall_dt * scale)
    return obs, lab, weight, valid, marked, gates, attempts, row_dt


def build_windows(runs, history: int, target_dt: float = 0.0, chunk: int = 1):
    """Stack per-run rows into (N, history, F) windows ending at each valid row.

    ``target_dt`` is the wall-clock spacing the deployed policy will see between
    successive observations — one control-loop iteration. Runs logged faster
    than that are sampled with a stride so every window covers the same span of
    time. Without it a 48 Hz coach log and a 10 Hz seed log produce windows
    covering 0.66 s and 3.2 s respectively, and the model is asked to learn
    from both as if they were the same thing.
    """
    xs, ys, ws, gs = [], [], [], []
    for obs, lab, weight, valid, _marked, gates, attempts, row_dt in runs:
        n = len(obs)
        stride = 1
        if target_dt > 0.0 and row_dt > 0.0:
            stride = max(1, int(round(target_dt / row_dt)))
        # First row index of the attempt each row belongs to. A sim reset
        # teleports the drone, so a window may not reach back across one.
        seg_start = np.zeros(n, dtype=np.int64)
        for k in range(1, n):
            seg_start[k] = k if attempts[k] != attempts[k - 1] else seg_start[k - 1]
        # Same one-shot visual snap the planner applies in flight, so a
        # window ending on a new lock is not 63 frames of the previous gate.
        live = obs.copy()
        for end in range(n):
            if (
                SNAP_VISUAL
                and end > 0
                and attempts[end] == attempts[end - 1]
                and visual_target_changed(live[end - 1], live[end])
            ):
                lo_snap = max(
                    int(seg_start[end]),
                    end - (history - 1) * stride,
                )
                apply_visual_snap(live[lo_snap:end], live[end])
            if not valid[end]:
                continue
            lo = int(seg_start[end])
            idx = [max(lo, end - stride * i) for i in range(history - 1, -1, -1)]
            window = live[idx]
            if chunk > 1:
                # Targets for this step and the next chunk-1, spaced like the
                # history. Clamp at the end of the attempt rather than bleeding
                # into the next one.
                fut = [min(n - 1, end + stride * j) for j in range(chunk)]
                if any(attempts[f] != attempts[end] for f in fut):
                    continue
                target = lab[fut]
            else:
                target = lab[end]
            xs.append(window)
            ys.append(target)
            ws.append(weight[end])
            gs.append(gates[end])
    if not xs:
        return None
    return (
        np.stack(xs).astype(np.float32),
        np.stack(ys).astype(np.float32),
        np.asarray(ws, dtype=np.float32),
        np.asarray(gs, dtype=np.int32),
    )


def augment_batch(xb, rng, *, jitter_px: float, dropout: float,
                  frame_w: float = FRAME_W, frame_h: float = FRAME_H):
    """Vectorised corner jitter and dropout over a whole batch of windows.

    Equivalent to calling ``race_obs.augment_corners`` on every frame, but as
    array operations. The scalar version stays the reference implementation and
    the two are checked against each other in the tests.
    """
    import torch

    arr = xb.detach().cpu().numpy().copy()
    n_kp = KEYPOINT_COUNT
    u_idx = np.arange(0, 2 * n_kp, 2)
    v_idx = u_idx + 1
    vis_idx = np.arange(2 * n_kp, 3 * n_kp)

    vis = arr[..., vis_idx] > 0.0
    # Drop a visible corner outright.
    drop = (rng.random(vis.shape) < dropout) & vis
    # Jitter whatever survives.
    keep = vis & ~drop
    du = rng.normal(0.0, jitter_px, size=vis.shape) / frame_w
    dv = rng.normal(0.0, jitter_px, size=vis.shape) / frame_h

    u = arr[..., u_idx]
    v = arr[..., v_idx]
    u = np.where(keep, np.clip(u + du, 0.0, 1.0), u)
    v = np.where(keep, np.clip(v + dv, 0.0, 1.0), v)
    u = np.where(drop, NOT_SEEN, u)
    v = np.where(drop, NOT_SEEN, v)
    arr[..., u_idx] = u
    arr[..., v_idx] = v
    arr[..., vis_idx] = np.where(drop, 0.0, arr[..., vis_idx])
    return torch.from_numpy(arr)


def channel_scales(Y: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Per-channel spread, for making the L1 loss scale-free across outputs.

    Without this the loss is dominated by whichever command happens to be
    largest in raw units, and a *sparse* channel is simply averaged away: an L1
    loss fits the conditional median, so a channel that is zero in 98% of frames
    is best predicted by always emitting zero. That is exactly what happened to
    yaw here -- it is non-zero in 1.9% of frames, the trained policy emitted
    |yaw| < 0.01 rad/s in flight, and the drone could not turn its nose even
    though the human demonstrations clearly did.

    Scaling each channel by its mean absolute deviation restores the rare,
    large yaw bursts to comparable weight against the ever-present thrust term.
    """
    med = np.median(Y, axis=0, keepdims=True)
    mad = np.mean(np.abs(Y - med), axis=0)
    return np.maximum(mad, floor).astype(np.float32)


def gate_balance_weights(
    gates: np.ndarray,
    *,
    clip: float = 4.0,
) -> np.ndarray:
    """Inverse-frequency weight per active gate, clipped.

    Long runs pile up enormous numbers of frames on whichever gate the drone
    loitered in front of -- measured at 68% of the whole corpus sitting on gate
    0 -- and unweighted that simply trains a start-gate specialist. Weighting
    rather than discarding keeps the rare late-gate frames, and the clip stops a
    gate with a handful of samples from dominating instead.
    """
    out = np.ones(len(gates), dtype=np.float32)
    present = [g for g in np.unique(gates) if g >= 0]
    if len(present) < 2:
        return out
    counts = {int(g): int(np.sum(gates == g)) for g in present}
    target = float(np.median(list(counts.values())))
    for g, count in counts.items():
        if count <= 0:
            continue
        factor = target / float(count)
        factor = float(np.clip(factor, 1.0 / clip, clip))
        out[gates == g] = factor
    return out


def warm_start_policy(model, path: Path, device: str):
    """Load ``path`` into ``model``. Arch must match or this is a no-op crash.

    From-scratch DAgger on mixed seed+coach is what forgot the launch in r1.
    Warm-start keeps the seed's pitch/roll and edits recoveries on top.
    """
    from policy_net import load_policy

    loaded, blob = load_policy(path, map_location=device)
    want, got = model.arch(), loaded.arch()
    if want != got:
        raise SystemExit(
            f'--init {path} arch {got} does not match this train {want}'
        )
    model.load_state_dict(loaded.state_dict())
    return blob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--telem', type=Path, nargs='*', default=[])
    ap.add_argument(
        '--glob',
        action='append',
        default=None,
        help='file glob; pass more than once to mix seed and coach folders',
    )
    ap.add_argument('--history', type=int, default=DEFAULT_HISTORY)
    ap.add_argument(
        '--target-dt', type=float, default=0.02,
        help='sim seconds between observations at deployment (1 / --hz). '
             '0.2x seed logs are converted from wall time first. Default 0.02 '
             'matches a 50 Hz flyer so a short look-up after pitch-down is '
             'not smeared. 0 disables striding.',
    )
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument(
        '--save-last', action='store_true',
        help='write the final epoch, not the best validation snapshot. Use '
             'this to memorize a small seed (val early-stop is why a 400-epoch '
             'run can still fly unlike any demo).',
    )
    ap.add_argument('--lead-s', type=float, default=0.5,
                    help='back-date interventions by this many seconds')
    ap.add_argument('--tail-s', type=float, default=0.5,
                    help='keep this much recovery after control returns')
    ap.add_argument('--intervention-weight', type=float, default=3.0)
    ap.add_argument(
        '--drop-collision-s', type=float, default=0.0,
        help='invalidate rows within this many seconds of a logged contact '
             '(reads the run\'s events_*.txt). Try 1.0 for a clean BC seed.',
    )
    ap.add_argument(
        '--min-impulse', type=float, default=0.0,
        help='ignore contacts below this impulse when using --drop-collision-s',
    )
    ap.add_argument(
        '--balance-gates', action='store_true',
        help='inverse-frequency weight per active_gate, so a long loiter in '
             'front of one gate cannot dominate the loss',
    )
    ap.add_argument(
        '--no-channel-balance', dest='channel_balance',
        action='store_false', default=True,
        help='disable per-channel loss scaling (see channel_scales)',
    )
    ap.add_argument(
        '--keep-policy-frames', dest='drop_policy_frames',
        action='store_false', default=True,
        help='train on policy-flown frames too. Off by default: their labels '
             'are the policy\'s own commands, so learning from them is '
             'self-imitation, not imitation of you.',
    )
    ap.add_argument(
        '--loss', choices=('huber', 'l1', 'l2'), default='huber',
        help='huber (default) fits the conditional mean for typical errors '
             'while staying robust to human command spikes; pure l1 fits the '
             'median, which on these zero-inflated rate commands collapses to '
             '"do nothing" (measured: 76%% of policy frames under 0.05 rad/s)',
    )
    ap.add_argument('--augment', action='store_true', default=True)
    ap.add_argument('--no-augment', dest='augment', action='store_false')
    ap.add_argument('--jitter-px', type=float, default=10.0)
    ap.add_argument('--dropout', type=float, default=0.10)
    ap.add_argument('--sort-by-u', action='store_true',
                    help="use paper1's u-sorted corner order instead of identity")
    ap.add_argument(
        '--context', action='store_true',
        help='add course context (active_gate one-hot + lap progress) to the '
             'observation. Deliberately course-specific: it makes the problem '
             '"what do I do at gate 7 of THIS track" rather than a general gate '
             'follower, at the cost of portability to another course.',
    )
    ap.add_argument(
        '--velocity', action='store_true', default=True,
        help='add commanded body velocity (vx, vy, vz). Reconstructed from '
             'thrust + attitude + drag — not the accelerometer, not EKF pos. '
             'Default on; a new seed must be trained, old n_in=29/48 '
             'checkpoints still fly without it.',
    )
    ap.add_argument(
        '--no-velocity', dest='velocity', action='store_false',
        help='omit commanded velocity (match a pre-velocity checkpoint).',
    )
    ap.add_argument(
        '--chunk', type=int, default=1,
        help='predict this many future action steps per inference (action '
             'chunking). >1 forces a coherent short-term plan and lets the '
             'planner average overlapping predictions.',
    )
    ap.add_argument(
        '--bins', type=int, default=0,
        help='discretise each action channel into this many bins and classify '
             'instead of regress. The human stick is multi-modal (roll is '
             'parked 71%% of frames and slammed 18%%), and a regressor averages '
             'those modes into a permanent mild input. 0 = regression.',
    )
    ap.add_argument('--out', type=Path, default=ROOT / 'models' / 'policy.pt')
    ap.add_argument(
        '--init', type=Path, default=None,
        help='warm-start from this checkpoint. Arch (H/chunk/bins/n_in) must '
             'match. Use this for DAgger rounds so the seed launch is not '
             're-rolled from scratch.',
    )
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    paths = list(args.telem)
    for pattern in args.glob or []:
        paths += [Path(p) for p in globlib.glob(pattern)]
    paths = [p for p in dict.fromkeys(paths) if p.is_file()]
    if not paths:
        print('no telemetry files found — pass --telem or --glob', flush=True)
        return

    print(f'loading {len(paths)} run(s)', flush=True)
    runs = []
    total_rows = human_rows = 0
    for p in paths:
        loaded = load_run(
            p, lead_s=args.lead_s, tail_s=args.tail_s, sort_by_u=args.sort_by_u,
            drop_collision_s=args.drop_collision_s,
            min_impulse=args.min_impulse,
            drop_policy_frames=args.drop_policy_frames,
            with_context=args.context,
            with_velocity=args.velocity,
        )
        if loaded is None:
            print(f'  {p.name}: empty, skipped', flush=True)
            continue
        obs, lab, weight, valid, marked, _gates, _att, _dt = loaded
        runs.append(loaded)
        total_rows += len(obs)
        human_rows += int(marked.sum())
        extra = ''
        if args.drop_collision_s > 0.0:
            hits = collision_times(p, args.min_impulse)
            extra = f', {len(hits)} contact(s)'
        print(f'  {p.name}: {len(obs)} rows, {int(valid.sum())} usable, '
              f'{int(marked.sum())} in intervention windows, '
              f'dt={_dt:.3f}s sim{extra}')

    if not runs:
        print('nothing loaded', flush=True)
        return

    built = build_windows(
        runs, args.history, target_dt=args.target_dt, chunk=args.chunk
    )
    if built is None:
        print('no usable windows — is a gate ever visible in these runs?', flush=True)
        return
    X, Y, W, G = built
    # Apply the intervention weight now that the hint (1.0 / 2.0) is known.
    W = np.where(W > 1.0, args.intervention_weight, 1.0).astype(np.float32)
    n_feat = X.shape[2]
    print(f'\n{len(X)} windows of {args.history} frames x {n_feat} features', flush=True)
    print(f'intervention-weighted windows: {int((W > 1.0).sum())} '
          f'({100.0 * (W > 1.0).mean():.1f}%)')

    print('\nwindows per gate:', flush=True)
    balance = gate_balance_weights(G) if args.balance_gates else None
    for g in sorted({int(v) for v in G if v >= 0}):
        count = int(np.sum(G == g))
        share = 100.0 * count / len(G)
        factor = float(balance[G == g][0]) if balance is not None else 1.0
        note = f'  x{factor:.2f}' if balance is not None else ''
        print(f'  gate {g:>2}: {count:6d} ({share:5.1f}%){note}', flush=True)
    if balance is not None:
        W = (W * balance).astype(np.float32)
    else:
        top = max(
            (int(np.sum(G == g)) for g in {int(v) for v in G if v >= 0}),
            default=0,
        )
        if top > 0.4 * len(G):
            print(
                f'  NOTE: one gate holds {100.0 * top / len(G):.0f}% of all '
                'windows. Pass --balance-gates or the policy will specialise '
                'in it.'
            , flush=True)
    if total_rows and human_rows == 0:
        print('NOTE: no human interventions present — this is the Phase 5 seed '
              'run. Expect the resulting policy to fly poorly by design.', flush=True)

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from policy_net import RacePolicy, save_policy

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Split by window index, but shuffle first so both halves span the runs.
    idx = rng.permutation(len(X))
    if args.save_last or args.val_frac <= 0.0:
        n_val = 0
        train_idx, val_idx = idx, idx[: min(32, len(idx))]
    else:
        n_val = max(1, int(args.val_frac * len(X)))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}  train={len(train_idx)}  val={len(val_idx)}', flush=True)

    Xtr = torch.from_numpy(X[train_idx])
    Ytr = torch.from_numpy(Y[train_idx])
    Wtr = torch.from_numpy(W[train_idx])
    Xva = torch.from_numpy(X[val_idx]).to(device)
    Yva = torch.from_numpy(Y[val_idx]).to(device)

    loader = DataLoader(
        TensorDataset(Xtr, Ytr, Wtr), batch_size=args.batch, shuffle=True
    )
    flat_train = Y[train_idx].reshape(-1, LABEL_DIM)
    scales = (
        channel_scales(flat_train) if args.channel_balance
        else np.ones(LABEL_DIM, dtype=np.float32)
    )
    scale_t = torch.from_numpy(scales).to(device)
    print('channel loss scales (mean abs deviation): ' + '  '.join(
        f'{n}={s:.3f}' for n, s in zip(LABEL_NAMES, scales)
    ), flush=True)

    model = RacePolicy(
        n_in=n_feat, history=args.history, chunk=args.chunk, bins=args.bins
    ).to(device)
    if args.init is not None:
        warm_start_policy(model, args.init, device)
        print(f'warm-start from {args.init}', flush=True)

    # Bin targets for the categorical head. Y is (N, LABEL_DIM) or
    # (N, chunk, LABEL_DIM) in raw units; Yb mirrors it with bin indices.
    Yb_tr = Yb_va = None
    centers_t = None
    if args.bins:
        edges_lo = np.array(
            [ACTION_RANGES[n][0] for n in LABEL_NAMES], dtype=np.float32
        )
        edges_hi = np.array(
            [ACTION_RANGES[n][1] for n in LABEL_NAMES], dtype=np.float32
        )
        frac = (Y - edges_lo) / np.maximum(1e-9, edges_hi - edges_lo)
        Yb = np.clip((frac * args.bins).astype(np.int64), 0, args.bins - 1)
        if Yb.ndim == 2:
            Yb = Yb[:, None, :]
        Yb_tr = torch.from_numpy(Yb[train_idx])
        Yb_va = torch.from_numpy(Yb[val_idx]).to(device)
        centers = np.stack([
            np.asarray(bin_centers(args.bins)[n], dtype=np.float32)
            for n in LABEL_NAMES
        ])  # (LABEL_DIM, bins)
        centers_t = torch.from_numpy(centers).to(device)
        occupied = len(np.unique(Yb.reshape(-1, LABEL_DIM)[:, 1]))
        print(f'action bins={args.bins}  roll occupies {occupied} distinct bins',
              flush=True)
        loader = DataLoader(
            TensorDataset(Xtr, Ytr, Wtr, Yb_tr),
            batch_size=args.batch, shuffle=True,
        )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = None
    if not args.save_last:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_mae = float('inf')
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        seen = 0
        for batch in loader:
            if args.bins:
                xb, yb, wb, ybin = batch
                ybin = ybin.to(device)
            else:
                xb, yb, wb = batch
                ybin = None
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            if args.augment:
                # Corner randomisation, paper1 section 4.1. Vectorised: the
                # original looped in Python over every row of every window,
                # which at H=64 is millions of calls per epoch and dominated
                # training time.
                xb = augment_batch(
                    xb, rng, jitter_px=args.jitter_px, dropout=args.dropout,
                ).to(device)
            pred = model(xb)
            if args.bins:
                # Cross-entropy over action bins, per channel. No averaging
                # across modes: the model is asked which bin, not what value.
                logits = pred.permute(0, 3, 1, 2)  # (B, bins, chunk, LABEL_DIM)
                ce = torch.nn.functional.cross_entropy(
                    logits, ybin, reduction='none',
                )  # (B, chunk, LABEL_DIM)
                w = wb.view(-1, 1, 1)
                loss = (w * ce).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                run_loss += float(loss.detach()) * len(xb)
                seen += len(xb)
                continue
            # Residuals are divided by the per-channel spread so no single
            # command can swamp the others (see channel_scales); the sample
            # weight carries the intervention emphasis.
            resid = (pred - yb) / scale_t
            if args.loss == 'l1':
                # Fits the conditional median. On rate commands that are zero
                # in most frames that median is zero, which is why the first
                # seed policy coasted instead of steering.
                per = resid.abs()
            elif args.loss == 'l2':
                per = resid ** 2
            else:
                # Huber: quadratic (mean-seeking) for ordinary errors, linear
                # in the tails so a human's spike cannot dominate a batch.
                per = torch.nn.functional.huber_loss(
                    pred / scale_t, yb / scale_t,
                    reduction='none', delta=1.0,
                )
            # per is (batch, LABEL_DIM) or (batch, chunk, LABEL_DIM).
            w = wb.view(-1, *([1] * (per.dim() - 1)))
            loss = (w * per).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += float(loss.detach()) * len(xb)
            seen += len(xb)
        if sched is not None:
            sched.step()

        model.eval()
        with torch.no_grad():
            va = model(Xva)
            if args.bins:
                # Decode the argmax bin to its centre value, so validation is
                # measured on the action that would actually be flown.
                idx = va[:, 0, :, :].argmax(dim=-1)          # (B, LABEL_DIM)
                va_now = torch.gather(
                    centers_t.unsqueeze(0).expand(idx.shape[0], -1, -1),
                    2, idx.unsqueeze(-1),
                ).squeeze(-1)
            else:
                va_now = va[:, 0, :] if va.dim() == 3 else va
            # Validation stays in raw units so it remains comparable across
            # runs and against the mean-predictor baseline. With chunking, score
            # only the first predicted step -- that is the one actually flown.
            y_now = Yva[:, 0, :] if Yva.dim() == 3 else Yva
            mae = float((va_now - y_now).abs().mean())
            per_ch = (va_now - y_now).abs().mean(dim=0).tolist()
            yaw_span = float(va_now[:, 3].abs().max())
            # "Coasting": how often the policy emits a near-zero steering
            # command, against how often the human actually did. A model that
            # collapses to the median sits far above the human's rate here and
            # flies straight past gates doing nothing.
            idle_pred = float((va_now[:, 1].abs() < 0.05).float().mean())
            idle_true = float((y_now[:, 1].abs() < 0.05).float().mean())
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            # flush: stdout is block-buffered when piped to a file or Tee-Object,
            # so without this a long run shows no progress and looks hung.
            print(f'  epoch {epoch:>3}  train={run_loss / max(seen, 1):.4f}  '
                  f'val_mae={mae:.4f}  '
                  f'[thr {per_ch[0]:.3f} r {per_ch[1]:.3f} '
                  f'p {per_ch[2]:.3f} y {per_ch[3]:.3f}]  '
                  f'best={best_mae:.4f}  max|yaw|={yaw_span:.2f}  '
                  f'idle roll pred/human={100 * idle_pred:.0f}%/'
                  f'{100 * idle_true:.0f}%',
                  flush=True)

    if best_state is not None and not args.save_last:
        model.load_state_dict(best_state)
    elif args.save_last:
        print('saving last epoch (--save-last); not the best val snapshot', flush=True)

    # Baseline: predict the training mean. A policy that cannot beat this has
    # learned nothing, which is the check the earlier attempt lacked.
    Y_now = Y[:, 0, :] if Y.ndim == 3 else Y
    mean_pred = torch.from_numpy(
        Y_now[train_idx].mean(axis=0, keepdims=True)
    ).to(device)
    Yva_now = Yva[:, 0, :] if Yva.dim() == 3 else Yva
    with torch.no_grad():
        base_mae = float((mean_pred - Yva_now).abs().mean())
    print(f'\nbest val_mae={best_mae:.4f}   mean-predictor baseline={base_mae:.4f}', flush=True)
    if base_mae > 0:
        print(f'improvement over baseline: {base_mae / max(best_mae, 1e-9):.2f}x', flush=True)
    if best_mae >= base_mae:
        print('WARNING: no better than predicting the mean — do not fly this.', flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_policy(args.out, model.cpu(), extra={
        'val_mae': best_mae,
        'baseline_mae': base_mae,
        'history': args.history,
        'sort_by_u': bool(args.sort_by_u),
        'context': bool(args.context),
        'velocity': bool(args.velocity),
        'chunk': int(args.chunk),
        'bins': int(args.bins),
        'target_dt': float(args.target_dt),
        'runs': [str(p) for p in paths],
        'n_windows': int(len(X)),
        'intervention_windows': int((W > 1.0).sum()),
        'init': str(args.init) if args.init is not None else None,
    })
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
