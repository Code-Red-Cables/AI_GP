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

Deliberately excluded, each because it is unavailable or misleading here:

  position / velocity   VQ2 publishes neither; the EKF's belief diverged to
                        1e7 m, so it is not a substitute
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
from typing import Iterable, Optional, Sequence

FRAME_W = 640.0
FRAME_H = 360.0

KEYPOINT_COUNT = 8
NOT_SEEN = -1.0

# Collisions spike the IMU well past anything flyable; the plant's rate command
# caps near 2.8 rad/s. Real flight passes through untouched.
GYRO_CLIP = 8.0

CORNER_CHANNELS = tuple(
    f'{axis}{i}' for i in range(KEYPOINT_COUNT) for axis in ('u', 'v')
)
VIS_CHANNELS = tuple(f'vis{i}' for i in range(KEYPOINT_COUNT))
STATE_CHANNELS = ('roll', 'pitch', 'gx', 'gy', 'gz')

FEATURE_NAMES: tuple[str, ...] = CORNER_CHANNELS + VIS_CHANNELS + STATE_CHANNELS
FEATURE_DIM = len(FEATURE_NAMES)

LABEL_NAMES = ('thrust', 'roll_rate', 'pitch_rate', 'yaw_rate')
LABEL_DIM = len(LABEL_NAMES)

# Paper1 Table 5 / section 4.2: H=4 gives 0% success, H=8 gives 84%, H=32
# gives 100%. The earlier attempt in this repo used H=1.
DEFAULT_HISTORY = 32


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
    min_confidence: float = 0.25,
    sort_by_u: bool = False,
    frame_w: float = FRAME_W,
    frame_h: float = FRAME_H,
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
        seen = (
            math.isfinite(u) and math.isfinite(v)
            and not (u == 0.0 and v == 0.0)
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

    return corners + vis + state


def observation_from_row(row: dict, *, sort_by_u: bool = False) -> list[float]:
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

    return build_observation(
        kps,
        confs,
        roll=_num(row.get('roll')),
        pitch=_num(row.get('pitch')),
        gx=_num(row.get('gx_imu')),
        gy=_num(row.get('gy_imu')),
        gz=_num(row.get('gz_imu')),
        sort_by_u=sort_by_u,
    )


def labels_from_row(row: dict) -> list[float]:
    """The commanded action recorded for this row.

    These are the *logged* cmd_* values, i.e. AFTER controller.py applied
    RATE_SIGN_*. Anything replaying them has to divide the signs back out or the
    round trip inverts pitch.
    """
    return [
        _num(row.get('cmd_thrust')),
        _num(row.get('cmd_roll_rate')),
        _num(row.get('cmd_pitch_rate')),
        _num(row.get('cmd_yaw_rate')),
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
    if len(seq) >= history:
        return seq[-history:]
    return [list(seq[0]) for _ in range(history - len(seq))] + seq
