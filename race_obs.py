"""The observation vector for the HG-DAgger racing policy, in one place.

Both the training script and the flight planner import this. They have to: if
the two ever build the vector differently -- a different normalisation, a
different channel order, a different not-seen fill -- the policy is fed
something it never trained on and flies badly for a reason that is invisible in
the training metrics. That exact failure already happened once in this repo.

Representation follows paper1's corner input (Xing et al., CoRL 2024): the
policy sees only what a camera can produce, so it stays valid on VQ2 where no
privileged state exists.

  * 8 gate keypoints as normalised pixels in [0, 1]
  * a keypoint not seen is filled with NOT_SEEN (-1.0), which is out of band
    for [0, 1] and therefore unambiguous
  * a per-keypoint visibility flag, so the network never has to infer
    "is -1 a position or a sentinel"
  * roll and pitch (gravity-referenced, available on both builds)
  * body rates from HIGHRES_IMU
  * optional body velocity from commanded thrust + attitude + drag
    (``--velocity``). A quadrotor is its own accelerometer: world accel is
    ``g − (thr/hover)·g·(R @ e_z) − drag(v)``, integrated in FRD so yaw is
    never required. This is not privileged state and does not use the
    competition accelerometer (horizontal correlation with true velocity ~0)
    or the EKF position that once diverged to 1e7 m.

Deliberately excluded, each because it is unavailable or misleading here:

  position / EKF vel    VQ2 publishes neither; IMU-integrated EKF vel is not
                        a substitute. Commanded body velocity is the replacement.
  absolute yaw          unobservable without a magnetometer
  PnP range / pose      paper1 uses no pose estimation at all, and PnP was
                        non-null in 1 of 39 laps on this detector

DEVIATION FROM THE PLAN, stated openly. The plan says "corners sorted by u",
which is what paper1 does. We default to keypoint-identity order instead,
because our detector is an eight-keypoint two-ring model: ids 0-3 are the 2.7 m
outer square and 4-7 the 1.5 m opening. Sorting all eight by u interleaves the
rings and destroys which ring a point belongs to -- and the two rings having
different physical sizes is exactly what encodes apparent scale, hence range.
Paper1 sorted because its detector returned four unidentified corners of a
single ring, so sorting cost it nothing. Set sort_by_u=True to reproduce the
paper's scheme exactly.
"""
from __future__ import annotations

import math
import os
from typing import Iterable, Optional, Sequence

FRAME_W = 640.0
FRAME_H = 360.0

KEYPOINT_COUNT = 8
NOT_SEEN = -1.0

# How far outside the image a predicted keypoint may sit and still be believed,
# as a fraction of the frame dimension. A partially visible gate genuinely puts
# corners just past the border and the detector locates those well; a prediction
# hundreds of pixels outside is fabrication.
OFF_FRAME_MARGIN = float(os.environ.get('OBS_OFF_FRAME_MARGIN', '0.15'))

# Collisions spike the IMU well past anything flyable; the plant's rate command
# caps near 2.8 rad/s. Real flight passes through untouched.
GYRO_CLIP = 8.0

CORNER_CHANNELS = tuple(
    f'{axis}{i}' for i in range(KEYPOINT_COUNT) for axis in ('u', 'v')
)
VIS_CHANNELS = tuple(f'vis{i}' for i in range(KEYPOINT_COUNT))
STATE_CHANNELS = ('roll', 'pitch', 'gx', 'gy', 'gz')
VEL_CHANNELS = ('vx', 'vy', 'vz')
VEL_DIM = len(VEL_CHANNELS)
# Body-frame m/s. Race speed is ~20 m/s; clip collisions the way gyro is clipped.
VEL_CLIP = 20.0

FEATURE_NAMES: tuple[str, ...] = CORNER_CHANNELS + VIS_CHANNELS + STATE_CHANNELS
FEATURE_DIM = len(FEATURE_NAMES)
FEATURE_NAMES_VEL: tuple[str, ...] = FEATURE_NAMES + VEL_CHANNELS
FEATURE_DIM_VEL = len(FEATURE_NAMES_VEL)

# Observation layout. Visual target (corners + visibility + optional one-hot)
# can snap on a new lock; IMU (roll/pitch/gyro) and commanded velocity must not.
VISUAL_END = 2 * KEYPOINT_COUNT + KEYPOINT_COUNT  # 24
STATE_END = VISUAL_END + len(STATE_CHANNELS)  # 29; velocity, if present, follows
# A new lock is a centre jump of this many normalised pixels, a reacquire
# after unseen, or an active_gate one-hot change. Tracking jitter is smaller.
SNAP_VISUAL = os.environ.get('OBS_SNAP_VISUAL', '1').strip().lower() not in (
    '0', 'false', 'no',
)
SNAP_CENTER_DIST = float(os.environ.get('OBS_SNAP_CENTER_DIST', '0.20'))

# ---- optional course context ------------------------------------------------
# Which gate the sim says we are flying to, one-hot, plus fractional progress
# through the lap. Both come from race_status, which is ordinary onboard
# telemetry available at run time -- not privileged state.
#
# This is deliberately course-specific. It turns "learn a general gate-following
# controller" into "learn what to do at gate 7 of this track", which is a much
# easier function to fit from a handful of laps. It costs portability to another
# course, so enable it only when a single course is the goal.
N_GATES = 18
CONTEXT_CHANNELS: tuple[str, ...] = (
    tuple(f'gate{i}' for i in range(N_GATES)) + ('gate_frac',)
)
FEATURE_NAMES_CTX: tuple[str, ...] = FEATURE_NAMES + CONTEXT_CHANNELS
FEATURE_DIM_CTX = len(FEATURE_NAMES_CTX)
FEATURE_NAMES_VEL_CTX: tuple[str, ...] = FEATURE_NAMES_VEL + CONTEXT_CHANNELS
FEATURE_DIM_VEL_CTX = len(FEATURE_NAMES_VEL_CTX)


def feature_dim(with_context: bool = False, with_velocity: bool = False) -> int:
    n = FEATURE_DIM
    if with_velocity:
        n += VEL_DIM
    if with_context:
        n += len(CONTEXT_CHANNELS)
    return n


def state_end_of(obs) -> int:
    """Index after IMU (+ velocity). Context, if any, starts here."""
    n = len(_as_seq(obs))
    if n in (FEATURE_DIM_VEL, FEATURE_DIM_VEL_CTX):
        return STATE_END + VEL_DIM
    return STATE_END


def _as_seq(obs) -> list[float]:
    return list(obs)


def _visible_uv(obs, *, ids: Optional[Iterable[int]] = None) -> tuple[list[float], list[float]]:
    seq = _as_seq(obs)
    if ids is None:
        ids = range(KEYPOINT_COUNT)
    us: list[float] = []
    vs: list[float] = []
    for i in ids:
        if seq[2 * KEYPOINT_COUNT + i] > 0.0:
            us.append(seq[2 * i])
            vs.append(seq[2 * i + 1])
    return us, vs


def visible_center(obs) -> Optional[tuple[float, float]]:
    """Mean (u, v) of visible corners, or None if the gate is unseen."""
    us, vs = _visible_uv(obs)
    if not us:
        return None
    return (sum(us) / len(us), sum(vs) / len(vs))


def visible_span(obs) -> Optional[float]:
    """Apparent size: hypot of the u/v ranges of visible corners.

    Prefers the outer ring (ids 0–3) when at least two of those are seen,
    because that ring's physical size is what encodes range.
    """
    us, vs = _visible_uv(obs, ids=range(4))
    if len(us) < 2:
        us, vs = _visible_uv(obs)
    if len(us) < 2:
        return None
    return math.hypot(max(us) - min(us), max(vs) - min(vs))


def approach_potential(obs) -> Optional[float]:
    """span × alignment. Grows only when the gate gets bigger *and* stays aimed.

    None if the gate is unseen. Image-diagonal normalisation so a corner lock
    is ~0 and a centered lock is ~1, times apparent size in [0, √2].
    """
    span = visible_span(obs)
    center = visible_center(obs)
    if span is None or center is None:
        return None
    offset = math.hypot(center[0] - 0.5, center[1] - 0.5)
    align = max(0.0, 1.0 - offset / math.sqrt(0.5))
    return float(span * align)


def context_gate(obs) -> Optional[int]:
    """Argmax of the active_gate one-hot, or None if context is absent/empty."""
    seq = _as_seq(obs)
    start = state_end_of(seq)
    if len(seq) < start + N_GATES:
        return None
    onehot = seq[start: start + N_GATES]
    if not any(v > 0.0 for v in onehot):
        return None
    return max(range(N_GATES), key=lambda i: onehot[i])


def visual_target_changed(
    prev,
    curr,
    *,
    center_dist: float = SNAP_CENTER_DIST,
) -> bool:
    """True when ``curr`` is a new visual/scorekeeper target vs ``prev``.

    IMU-only motion does not count. Losing the gate (seen → unseen) does not
    count: that would paint sentinels over the last good view.
    """
    g0, g1 = context_gate(prev), context_gate(curr)
    if g0 is not None and g1 is not None and g0 != g1:
        return True
    c0, c1 = visible_center(prev), visible_center(curr)
    if c0 is None and c1 is not None:
        return True
    if c0 is not None and c1 is not None:
        du = c0[0] - c1[0]
        dv = c0[1] - c1[1]
        if math.hypot(du, dv) >= float(center_dist):
            return True
    return False


def apply_visual_snap(frames, source) -> None:
    """Copy keypoints, visibility, and context from ``source`` onto ``frames``.

    Leaves roll / pitch / gyro on each frame untouched. In-place. ``frames``
    is an iterable of mutable 1-d sequences (lists or ndarray rows).
    """
    src = _as_seq(source)
    vis = src[:VISUAL_END]
    imu_end = state_end_of(src)
    ctx = src[imu_end:] if len(src) > imu_end else None
    for fr in frames:
        fr[:VISUAL_END] = vis
        end = state_end_of(fr)
        if ctx is not None and len(fr) > end:
            fr[end:] = ctx


def context_features(gate_index) -> list[float]:
    """One-hot active gate plus fractional lap progress."""
    out = [0.0] * (N_GATES + 1)
    g = _num(gate_index)
    if math.isnan(g):
        return out
    idx = int(max(0, min(N_GATES - 1, int(g))))
    out[idx] = 1.0
    out[N_GATES] = idx / float(max(1, N_GATES - 1))
    return out

LABEL_NAMES = ('thrust', 'roll_rate', 'pitch_rate', 'yaw_rate')
LABEL_DIM = len(LABEL_NAMES)

# ---- action discretisation --------------------------------------------------
# The human's stick is not continuously distributed: measured over every seed
# lap, roll is within 0.1 rad/s of zero in 71% of frames and beyond 2.5 rad/s in
# 17.6%, with little in between. He holds, then commits.
#
# That is a multi-modal action distribution, and no single-output regressor can
# represent it: a mean-seeking loss (L2/Huber) lands *between* the modes and
# rolls mildly forever, while a median-seeking loss (L1) collapses onto the zero
# mode and never commits. Both failures were observed in flight.
#
# Predicting *which bin* the action falls in, as a classification, sidesteps the
# problem -- the argmax lands on a mode instead of averaging across them.
ACTION_RANGES = {
    'thrust': (0.05, 0.70),
    'roll_rate': (-3.2, 3.2),
    'pitch_rate': (-3.2, 3.2),
    'yaw_rate': (-3.2, 3.2),
}


def bin_centers(n_bins: int) -> dict[str, list[float]]:
    """Centre value of each bin, per action channel."""
    out: dict[str, list[float]] = {}
    for name, (lo, hi) in ACTION_RANGES.items():
        width = (hi - lo) / float(n_bins)
        out[name] = [lo + width * (i + 0.5) for i in range(n_bins)]
    return out


# How many bins either side of the winning bin contribute to the decoded value.
# Zero reproduces a hard argmax.
#
# A hard argmax is degenerate in practice. Measured on policy_h64 in flight, it
# emitted cmd_roll_rate of exactly 0.0000 for every frame of a 26 s run (the
# idle bin holds the most probability mass on roll, so it always won) while
# pitch snapped to the outermost bin, +3.05 rad/s, and held it for a full
# second -- 175 deg of rotation, which tumbled the aircraft. Both channels
# collapsed onto a single bin centre and never produced an intermediate value.
#
# Averaging the *whole* distribution instead would undo the point of the
# categorical head: with mass at both +3.1 and -3.1 the mean is zero, which is
# the mode-averaging that made the regression policy coast. Averaging only
# within a window around the winner keeps the choice of mode while letting the
# value land between bin centres.
BIN_DECODE_WINDOW = int(os.environ.get('BIN_DECODE_WINDOW', '2'))


def decode_bin_probs(
    probs: Sequence[Sequence[float]],
    n_bins: int,
    *,
    window: Optional[int] = None,
) -> list[float]:
    """Decode per-channel bin probabilities to one value per channel.

    ``probs`` is (LABEL_DIM, n_bins) and need not be normalised. Picks the
    highest-probability bin per channel, then returns the probability-weighted
    mean of the bin centres within ``window`` bins of it.
    """
    if window is None:
        window = BIN_DECODE_WINDOW
    window = max(0, int(window))
    centers = bin_centers(n_bins)
    out: list[float] = []
    for c, name in enumerate(LABEL_NAMES):
        row = [float(p) for p in probs[c]]
        cen = centers[name]
        best = max(range(len(row)), key=lambda i: row[i])
        lo = max(0, best - window)
        hi = min(len(row) - 1, best + window)
        wsum = 0.0
        vsum = 0.0
        for i in range(lo, hi + 1):
            w = max(0.0, row[i])
            wsum += w
            vsum += w * cen[i]
        out.append(cen[best] if wsum <= 0.0 else vsum / wsum)
    return out


def action_to_bin(value, channel: str, n_bins: int) -> int:
    """Index of the bin containing ``value`` for ``channel`` (clamped)."""
    lo, hi = ACTION_RANGES[channel]
    v = _num(value)
    if math.isnan(v):
        v = lo if channel == 'thrust' else 0.0
    frac = (v - lo) / max(1e-9, hi - lo)
    return int(max(0, min(n_bins - 1, int(frac * n_bins))))


def bin_to_action(index: int, channel: str, n_bins: int) -> float:
    lo, hi = ACTION_RANGES[channel]
    width = (hi - lo) / float(n_bins)
    idx = int(max(0, min(n_bins - 1, index)))
    return lo + width * (idx + 0.5)


def labels_to_bins(labels: Sequence[float], n_bins: int) -> list[int]:
    return [
        action_to_bin(labels[i], name, n_bins)
        for i, name in enumerate(LABEL_NAMES)
    ]


def bins_to_labels(indices: Sequence[int], n_bins: int) -> list[float]:
    return [
        bin_to_action(int(indices[i]), name, n_bins)
        for i, name in enumerate(LABEL_NAMES)
    ]

# Paper1 Table 5 / section 4.2: H=4 gives 0% success, H=8 gives 84%, H=32
# gives 100%. The earlier attempt in this repo used H=1.
#
# 64 rather than paper1's 32 because this course blinds the camera for longer
# than that covers. Detection drops from 65% at level flight to 31% beyond 40
# degrees of pitch -- the camera is tilted up 20 degrees, so driving hard aims it
# below the horizon -- and measured blind stretches ran 5-6 s of wall time. At
# the 10 Hz observation rate, H=32 spans 3.2 s and cannot bridge a typical gap;
# H=64 spans 6.4 s and can. Existing checkpoints carry their own value.
DEFAULT_HISTORY = int(os.environ.get('OBS_HISTORY', '64'))


def _num(value) -> float:
    if value is None or value == '' or value == 'nan':
        return math.nan
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def build_observation(
    keypoints: Optional[Sequence[Sequence[float]]] = None,
    keypoint_confidences: Optional[Sequence[float]] = None,
    *,
    roll: float = 0.0,
    pitch: float = 0.0,
    gx: float = 0.0,
    gy: float = 0.0,
    gz: float = 0.0,
    vx: Optional[float] = None,
    vy: Optional[float] = None,
    vz: Optional[float] = None,
    with_velocity: bool = False,
    min_confidence: float = 0.25,
    sort_by_u: bool = False,
    frame_w: float = FRAME_W,
    frame_h: float = FRAME_H,
    gate_index=None,
) -> list[float]:
    """One observation row, ordered to match FEATURE_NAMES.

    ``keypoints`` is (8, 2) in pixels. A point is treated as unseen when it is
    non-finite, exactly (0, 0) (the detector's not-seen convention), or below
    ``min_confidence``.
    """
    pts: list[tuple[float, float, bool]] = []
    kp_list = list(keypoints) if keypoints is not None else []
    conf_list = list(keypoint_confidences) if keypoint_confidences is not None else []

    for i in range(KEYPOINT_COUNT):
        u = v = math.nan
        if i < len(kp_list):
            pair = kp_list[i]
            try:
                u, v = _num(pair[0]), _num(pair[1])
            except (TypeError, IndexError):
                u = v = math.nan
        conf = _num(conf_list[i]) if i < len(conf_list) else 1.0
        # A pose model predicts all eight keypoints whether or not they are in
        # frame. Clamping an off-screen prediction to the image edge and
        # flagging it "seen" tells the network a corner is at the border when it
        # is really a guess about something outside the picture -- observed in
        # flight as corners "pushed off to the side" as a gate fills the view.
        # Anything beyond the frame by more than OFF_FRAME_MARGIN of its
        # dimension is reported as not-seen instead.
        mx = OFF_FRAME_MARGIN * frame_w
        my = OFF_FRAME_MARGIN * frame_h
        in_frame = (-mx <= u <= frame_w + mx) and (-my <= v <= frame_h + my)
        seen = (
            math.isfinite(u) and math.isfinite(v)
            and not (u == 0.0 and v == 0.0)
            and in_frame
            and (math.isnan(conf) or conf >= min_confidence)
        )
        pts.append((u, v, seen))

    if sort_by_u:
        # paper1's scheme. Unseen points sort last so they do not shuffle the
        # visible ordering. See the module docstring for why this is not default.
        pts.sort(key=lambda p: (not p[2], p[0] if p[2] else math.inf))

    corners: list[float] = []
    vis: list[float] = []
    for u, v, seen in pts:
        if seen:
            corners.append(min(1.0, max(0.0, u / frame_w)))
            corners.append(min(1.0, max(0.0, v / frame_h)))
            vis.append(1.0)
        else:
            corners.append(NOT_SEEN)
            corners.append(NOT_SEEN)
            vis.append(0.0)

    state: list[float] = []
    for value in (roll, pitch):
        f = _num(value)
        state.append(0.0 if math.isnan(f) else f)
    for value in (gx, gy, gz):
        f = _num(value)
        state.append(0.0 if math.isnan(f) else _clip(f, GYRO_CLIP))
    if with_velocity:
        for value in (vx, vy, vz):
            f = _num(value)
            state.append(0.0 if math.isnan(f) else _clip(f, VEL_CLIP))

    if gate_index is None:
        return corners + vis + state
    return corners + vis + state + context_features(gate_index)


# Attitude channels in order of trust. ``att_raw`` is the sim's own ATTITUDE
# message; spec section 4.3 lists it as supported, but it has been measured
# absent on every build available here (0 of 5854 rows on a full 17-gate lap),
# so in practice ``ahrs`` is the working source. Both of those are
# gravity-referenced. ``ekf`` is integrated gyro, measured walking +4 deg to
# -23 deg over 50 s, and is NOT trusted for training: a policy fed it learns an
# attitude channel that is a slow random walk.
ATTITUDE_CHANNELS = (
    ('att_raw', 'att_raw_roll', 'att_raw_pitch'),
    ('ahrs', 'ahrs_roll', 'ahrs_pitch'),
    ('ekf', 'roll', 'pitch'),
)
TRUSTED_ATTITUDE_SOURCES = frozenset({'att_raw', 'ahrs'})


def attitude_source_of_row(row: dict) -> str:
    """Which attitude channel this row will actually supply."""
    for name, r_key, p_key in ATTITUDE_CHANNELS:
        if math.isfinite(_num(row.get(r_key))) and math.isfinite(
            _num(row.get(p_key))
        ):
            return name
    return 'none'


def attitude_is_trusted(row: dict) -> bool:
    """True when roll/pitch come from a gravity-referenced source.

    Rows falling through to the EKF belief are excluded from training rather
    than fed drift. In practice these are pre-arm rows, where the controller
    AHRS has not published yet and there is no useful action label anyway.
    """
    return attitude_source_of_row(row) in TRUSTED_ATTITUDE_SOURCES


def attitude_from_row(row: dict) -> tuple[float, float]:
    """Roll and pitch for the observation, most trustworthy source first.

    See ``ATTITUDE_CHANNELS``. Falls back to the EKF belief so old logs still
    parse; use ``attitude_is_trusted`` to decide whether to train on the row.
    """
    for _name, r_key, p_key in ATTITUDE_CHANNELS:
        r, p = _num(row.get(r_key)), _num(row.get(p_key))
        if math.isfinite(r) and math.isfinite(p):
            return r, p
    return math.nan, math.nan


def observation_from_row(
    row: dict,
    *,
    sort_by_u: bool = False,
    with_context: bool = False,
    with_velocity: bool = False,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
) -> list[float]:
    """Build an observation from one telemetry CSV row.

    Expects the ``kp{i}_u`` / ``kp{i}_v`` / ``kp{i}_c`` columns written by the
    logger. Falls back to the single gate centre when keypoints are absent, so
    older logs still parse -- but such rows carry only one usable point and
    should not be trained on.
    """
    kps: list[tuple[float, float]] = []
    confs: list[float] = []
    have_kp = f'kp0_u' in row
    if have_kp:
        for i in range(KEYPOINT_COUNT):
            kps.append((_num(row.get(f'kp{i}_u')), _num(row.get(f'kp{i}_v'))))
            confs.append(_num(row.get(f'kp{i}_c')))
    else:
        centre = (_num(row.get('gate_u')), _num(row.get('gate_v')))
        kps = [centre] + [(math.nan, math.nan)] * (KEYPOINT_COUNT - 1)
        confs = [_num(row.get('gate_conf'))] + [math.nan] * (KEYPOINT_COUNT - 1)

    roll, pitch = attitude_from_row(row)
    return build_observation(
        kps,
        confs,
        roll=roll,
        pitch=pitch,
        gx=_num(row.get('gx_imu')),
        gy=_num(row.get('gy_imu')),
        gz=_num(row.get('gz_imu')),
        vx=vx,
        vy=vy,
        vz=vz,
        with_velocity=with_velocity,
        sort_by_u=sort_by_u,
        gate_index=row.get('active_gate') if with_context else None,
    )


def commanded_velocity_from_rows(rows: Sequence[dict]) -> list[list[float]]:
    """Replay commanded body velocity over a telemetry log.

    Uses wall-clock ``t`` (physics on this plant tracks wall, not sim_boot).
    Zeros on a new ``attempt``. Same integrator the live policy steps.
    """
    import numpy as np
    from ekf.commanded_accel import BodyVelocityIntegrator

    try:
        import config as _cfg
        hover0 = float(getattr(_cfg, 'HOVER_THRUST', 0.255))
        k_body = np.array(
            [
                float(getattr(_cfg, 'DRAG_KX', -0.50)),
                float(getattr(_cfg, 'DRAG_KY', -0.50)),
                float(getattr(_cfg, 'DRAG_KZ', -0.15)),
            ],
            dtype=np.float64,
        )
    except Exception:
        hover0 = 0.255
        k_body = None

    integ = BodyVelocityIntegrator(hover_trim=hover0, k_body=k_body)
    out: list[list[float]] = []
    prev_t: Optional[float] = None
    prev_attempt: Optional[int] = None
    for row in rows:
        attempt = _num(row.get('attempt'))
        att_i = int(attempt) if math.isfinite(attempt) else 0
        if prev_attempt is not None and att_i != prev_attempt:
            integ.reset()
            prev_t = None
        t = _num(row.get('t'))
        if prev_t is None or not math.isfinite(t):
            dt = 0.02
        else:
            dt = t - prev_t
        if not math.isfinite(dt) or dt <= 0.0:
            dt = 0.02
        roll, pitch = attitude_from_row(row)
        hover = _num(row.get('hover_trim'))
        if math.isnan(hover):
            hover = _num(row.get('hover_thrust'))
        if math.isnan(hover):
            hover = hover0
        thrust = _num(row.get('cmd_thrust'))
        if math.isnan(thrust):
            thrust = hover
        omega = np.array(
            [_num(row.get('gx_imu')), _num(row.get('gy_imu')), _num(row.get('gz_imu'))],
            dtype=np.float64,
        )
        v = integ.step(dt, thrust, roll, pitch, omega, hover_trim=hover)
        out.append([float(v[0]), float(v[1]), float(v[2])])
        prev_t = t if math.isfinite(t) else prev_t
        prev_attempt = att_i
    return out


def labels_from_row(row: dict) -> list[float]:
    """The commanded action recorded for this row, as *requested* rates.

    Logger stores cmd_* AFTER ``controller.py`` applied ``RATE_SIGN_*``. The
    student emits pre-sign rates into the same kalman path, so training undoes
    those signs here. Thrust is unsigned and passes through unchanged.
    """
    import config as _cfg

    def _unsign(value, sign: float) -> float:
        v = _num(value)
        if math.isnan(v) or sign == 0.0:
            return v
        return v / float(sign)

    return [
        _num(row.get('cmd_thrust')),
        _unsign(row.get('cmd_roll_rate'), getattr(_cfg, 'RATE_SIGN_ROLL', 1.0)),
        _unsign(row.get('cmd_pitch_rate'), getattr(_cfg, 'RATE_SIGN_PITCH', 1.0)),
        _unsign(row.get('cmd_yaw_rate'), getattr(_cfg, 'RATE_SIGN_YAW', 1.0)),
    ]


def augment_corners(
    obs: list[float],
    rng,
    *,
    jitter_px: float = 10.0,
    dropout: float = 0.10,
    frame_w: float = FRAME_W,
    frame_h: float = FRAME_H,
) -> list[float]:
    """Paper1's corner randomisation: pixel jitter plus per-corner dropout.

    Section 4.1 uses plus/minus 10 px on a 1280x760 frame and a 10% chance of a
    corner going missing. Retune both against the deployed detector's measured
    noise -- training on cleaner corners than you will receive is the standard
    way this pipeline fails.
    """
    out = list(obs)
    n_corner = KEYPOINT_COUNT * 2
    for i in range(KEYPOINT_COUNT):
        vis_idx = n_corner + i
        if out[vis_idx] <= 0.0:
            continue
        if rng.random() < dropout:
            out[2 * i] = NOT_SEEN
            out[2 * i + 1] = NOT_SEEN
            out[vis_idx] = 0.0
            continue
        du = rng.normal(0.0, jitter_px) / frame_w
        dv = rng.normal(0.0, jitter_px) / frame_h
        out[2 * i] = min(1.0, max(0.0, out[2 * i] + du))
        out[2 * i + 1] = min(1.0, max(0.0, out[2 * i + 1] + dv))
    return out


def stack_history(
    rows: Iterable[list[float]],
    history: int = DEFAULT_HISTORY,
) -> list[list[float]]:
    """Left-pad a sequence of observations to exactly ``history`` frames.

    Padding repeats the oldest available frame rather than inserting zeros,
    because a zero row is a valid-looking observation with every corner at the
    image origin.
    """
    seq = list(rows)
    if not seq:
        return [[0.0] * FEATURE_DIM for _ in range(history)]
    width = len(seq[0])
    if width != FEATURE_DIM:
        # Context-augmented rows are wider; keep the pad the same shape.
        return (
            [list(seq[0]) for _ in range(max(0, history - len(seq)))]
            + seq[-history:]
        )
    if len(seq) >= history:
        return seq[-history:]
    return [list(seq[0]) for _ in range(history - len(seq))] + seq
