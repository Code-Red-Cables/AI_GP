"""Gate PnP: 4 ordered inner-square corner pixels + known gate size -> gate pose
in the camera frame, via cv2.solvePnPGeneric with SOLVEPNP_IPPE_SQUARE.

Corner ordering contract (shared with perception.detector):
  image corners are TL, TR, BR, BL of the gate's INNER square as seen upright.

Gate frame (matches frames.py docstring): x right along the face, y UP along
the face, z out of the plane toward the approaching drone.

OpenCV quirk (empirically established on 4.11, pinned by tests/test_pnp.py):
IPPE_SQUARE with the documented object-point order only returns the correct
zero-error solution when the image points are fed vertically flipped
(BL,BR,TR,BL order), and the pose it then returns is expressed in a gate frame
with y DOWN / z pointing AWAY from the camera. We therefore:
  1. reorder image points [BL, BR, TR, TL] before solving,
  2. convert each solution to OUR gate frame via R_ours = R_solver @ diag(1,-1,-1),
  3. rank solutions by OUR OWN reprojection error (OpenCV's reported errors
     were observed to be unreliable here), computed in our frame.
If a future OpenCV changes this behavior, the synthetic round-trip tests fail
loudly — do not "fix" this file without re-running them.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from aigp.geometry.frames import Pose

# 180-degree rotation about gate x: solver's implied frame <-> our gate frame
_R_FLIP = np.diag([1.0, -1.0, -1.0])
_IMG_REORDER = [3, 2, 1, 0]   # TL,TR,BR,BL -> BL,BR,TR,TL


def gate_object_points(size_m):
    """TL, TR, BR, BL of the square in OUR gate frame (x right, y up)."""
    h = size_m / 2.0
    return np.array([
        [-h, +h, 0.0],   # TL
        [+h, +h, 0.0],   # TR
        [+h, -h, 0.0],   # BR
        [-h, -h, 0.0],   # BL
    ], dtype=np.float64)


@dataclass
class PnPResult:
    T_cg: Pose             # gate pose in camera frame (our gate frame)
    reproj_err: float      # RMS pixels of the chosen solution (our computation)
    ambiguity: float       # err_best / err_second (small = confident)


def is_front_view(res: "PnPResult") -> bool:
    """True if the solved gate is seen from its FRONT face (gate +z, which
    points toward the approaching drone, has a component toward the camera).

    A square is symmetric: corner orderings that are mirrors/rotations of the
    truth are rigidly realizable as back/rolled views, so PnP alone cannot
    reject them — the caller (estimator/planner, which knows track context)
    must use this discriminator plus Mahalanobis gating.
    """
    return bool(res.T_cg.R[2, 2] < 0.0)


def _reproj_rms(K, R, t, obj, img):
    Xc = obj @ R.T + t
    if np.any(Xc[:, 2] <= 1e-6):
        return np.inf
    uv = Xc @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    return float(np.sqrt(np.mean(np.sum((uv - img) ** 2, axis=1))))


def solve_gate_pnp(corners_px, size_m, K, max_reproj_err=5.0) -> PnPResult | None:
    """corners_px: (4,2) TL,TR,BR,BL image points of the inner square.

    Solutions worse than max_reproj_err RMS pixels are rejected (a mislabeled
    or mirrored quad is not rigidly realizable and shows tens of px of error).
    """
    obj = gate_object_points(size_m)
    img = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    img_flipped = img[_IMG_REORDER].reshape(4, 1, 2)

    candidates = []
    try:
        n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj, img_flipped, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
    except cv2.error:
        n, rvecs, tvecs = 0, [], []
    for i in range(n):
        R_s, _ = cv2.Rodrigues(rvecs[i])
        t = np.asarray(tvecs[i]).flatten()
        if not (np.all(np.isfinite(R_s)) and np.all(np.isfinite(t))):
            continue
        R = R_s @ _R_FLIP   # solver frame -> our gate frame
        if t[2] <= 0:       # gate must be in front of the camera
            continue
        candidates.append((_reproj_rms(K, R, t, obj, img), R, t))

    candidates = [c for c in candidates if np.isfinite(c[0])]
    candidates.sort(key=lambda c: c[0])

    if candidates and candidates[0][0] <= max_reproj_err:
        err, R, t = candidates[0]
        second = candidates[1][0] if len(candidates) > 1 else np.inf
        ambiguity = err / second if np.isfinite(second) and second > 1e-9 else 0.0
        return PnPResult(T_cg=Pose(R, t), reproj_err=err, ambiguity=float(ambiguity))

    # IPPE_SQUARE has rare degenerate configurations (returns NaNs or garbage).
    # ITERATIVE with the direct (unflipped) correspondence recovers exactly.
    try:
        ok, rvec, tvec = cv2.solvePnP(
            obj, img.reshape(4, 1, 2), K, None, flags=cv2.SOLVEPNP_ITERATIVE
        )
    except cv2.error:
        return None
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec).flatten()
    if not (np.all(np.isfinite(R)) and np.all(np.isfinite(t))) or t[2] <= 0:
        return None
    err = _reproj_rms(K, R, t, obj, img)
    if not np.isfinite(err) or err > max_reproj_err:
        return None
    return PnPResult(T_cg=Pose(R, t), reproj_err=err, ambiguity=0.0)
