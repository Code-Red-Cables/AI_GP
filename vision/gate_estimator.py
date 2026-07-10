"""Gate geometry estimation (see PLAN.md 8.3).

Turns a 2D `GateDetection` (pixels) into a 3D estimate of where the gate is
relative to the drone, using the pinhole `camera_model`:

- **size method** (always available): assume the gate opening is a fronto-parallel
  1.5 m square; recover optical-axis depth from its apparent pixel size, then
  deproject the opening center to a camera-frame point.
- **PnP method** (when 4 corners are detected): `cv2.solvePnP` on the known
  1.5 m square gives a full metric pose (position + the gate's surface normal),
  independent of the fronto-parallel assumption. This is the "reverse solver".

Outputs a plain dict matching `shared_data['vision']` (PLAN 8.6). All vectors are
in metres. `gate_body` is the gate center relative to the drone in BODY axes
(x-fwd, y-right, z-down); `gate_ned` is the absolute world position and is only
filled when both attitude and drone NED position are supplied.
"""

import math
from dataclasses import dataclass

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

import camera_model as cm
from vision.gate_detector import GateDetection

_S = cm.GATE_INNER_M / 2.0
_GATE_OBJ_PTS = np.array(
    [[-_S, _S, 0.0], [_S, _S, 0.0], [_S, -_S, 0.0], [-_S, -_S, 0.0]],
    dtype=np.float32,
)

@dataclass
class GateObservation:
    ts_ns: int                    # sim_time_ns of the source frame
    frame_id: int
    gate_body: np.ndarray         # (3,) gate center, body NED frame
    normal_body: np.ndarray       # (3,) unit normal, body frame (sign: toward drone)
    gate_ned: np.ndarray | None   # (3,) world NED (needs pose at frame time)
    normal_ned: np.ndarray | None
    range_m: float
    method: str                   # 'pnp' | 'size'
    confidence: float             # carried from detection
    center_px: tuple[float, float]

def _opening_side_px(det: GateDetection) -> float:
    if det.corners_px is not None and len(det.corners_px) == 4:
        pts = [np.asarray(c, float) for c in det.corners_px]
        edges = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
        return float(np.mean(edges))
    return float(math.sqrt(max(det.area_px, 1.0)))

def _bearing(gate_body: np.ndarray) -> tuple[float, float]:
    x, y, z = gate_body[0], gate_body[1], gate_body[2]
    az = math.atan2(y, x)
    el = math.atan2(-z, math.hypot(x, y))
    return float(az), float(el)

def _solve_pnp(det: GateDetection):
    if cv2 is None or det.corners_px is None or len(det.corners_px) != 4:
        return None
    img_pts = np.array(det.corners_px, dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(
        _GATE_OBJ_PTS, img_pts, cm.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    gate_cam = tvec.reshape(3)
    normal_cam = R @ np.array([0.0, 0.0, 1.0])
    return gate_cam, normal_cam

def estimate_gates(dets: list[GateDetection],
                   attitude: dict | None,
                   position_ned: dict | None,
                   ts_ns: int = 0,
                   frame_id: int = 0) -> list[GateObservation]:
    obs_list = []
    
    for det in dets:
        method = "size"
        normal_body = None

        pnp = _solve_pnp(det)
        if pnp is not None:
            gate_cam, normal_cam = pnp
            gate_body = cm.cam_to_body(gate_cam)
            normal_body = cm.cam_to_body(normal_cam)
            # flip normal if it faces away from the drone (x > 0 means pointing forward, so flip it)
            # Actually, standard is normal faces towards drone. 
            # "flip incoming normals to the hemisphere facing the drone's side" is for mapper, 
            # here we just attach the normal.
            method = "pnp"
        else:
            side_px = _opening_side_px(det)
            Z = cm.range_from_size(side_px, real_size=cm.GATE_INNER_M, f=cm.FX)
            u, v = det.center_px
            gate_cam = cm.deproject(u, v, Z)
            gate_body = cm.cam_to_body(gate_cam)

        range_m = float(np.linalg.norm(gate_body))

        gate_ned = None
        normal_ned = None
        if attitude is not None:
            r, p, y = attitude["roll"], attitude["pitch"], attitude["yaw"]
            offset_ned = cm.body_to_ned(gate_body, r, p, y)
            if normal_body is not None:
                normal_ned = cm.body_to_ned(normal_body, r, p, y)
                
            if position_ned is not None:
                pos = (position_ned.get("x", 0.0), position_ned.get("y", 0.0), position_ned.get("z", 0.0)) if isinstance(position_ned, dict) else position_ned
                gate_ned = np.array([a + b for a, b in zip(pos, offset_ned)], dtype=np.float64)
            else:
                gate_ned = np.array(offset_ned, dtype=np.float64)

        # Apply confidence penalty for size method (from Phase 1 spec: "size method fallback gets confidence *= 0.7")
        conf = float(det.confidence)
        if method == "size":
            conf *= 0.7

        obs = GateObservation(
            ts_ns=ts_ns,
            frame_id=frame_id,
            gate_body=np.array(gate_body, dtype=np.float64),
            normal_body=np.array(normal_body, dtype=np.float64) if normal_body is not None else np.zeros(3),
            gate_ned=gate_ned,
            normal_ned=np.array(normal_ned, dtype=np.float64) if normal_ned is not None else None,
            range_m=range_m,
            method=method,
            confidence=conf,
            center_px=det.center_px
        )
        obs_list.append(obs)
        
    return obs_list

def estimate_gate(det, attitude=None, position_ned=None, use_pnp=True, ts=None):
    """Legacy wrapper for old tests."""
    if det is None:
        return {"ts": ts, "detected": False, "confidence": 0.0}
    obs_list = estimate_gates([det], attitude, position_ned, ts_ns=ts or 0)
    if not obs_list:
        return {"ts": ts, "detected": False, "confidence": 0.0}
    obs = obs_list[0]
    return {
        "ts": ts,
        "detected": True,
        "confidence": obs.confidence,
        "center_px": obs.center_px,
        "corners_px": det.corners_px,
        "area_px": det.area_px,
        "range_m": obs.range_m,
        "bearing": _bearing(obs.gate_body),
        "gate_body": tuple(float(c) for c in obs.gate_body),
        "gate_ned": tuple(float(c) for c in obs.gate_ned) if obs.gate_ned is not None else None,
        "normal_body": tuple(float(c) for c in obs.normal_body) if obs.normal_body is not None else None,
        "method": obs.method,
    }
