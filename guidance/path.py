import math
import numpy as np
from typing import List, Tuple, Optional

# Constants defaults (matching old config if cfg is missing)
SAMPLES_PER_SEG = 40
CURV_STENCIL_M = 3.5
FWD_WINDOW = 40
BACK_WINDOW = 8

def _catmull_rom_segment(p0, p1, p2, p3, n, alpha=0.5):
    def _knot(ti, pi, pj):
        d = float(np.linalg.norm(pj - pi))
        return ti + (d ** alpha if d > 1e-9 else 1e-9)

    t0 = 0.0
    t1 = _knot(t0, p0, p1)
    t2 = _knot(t1, p1, p2)
    t3 = _knot(t2, p2, p3)
    out = []
    for i in range(n):
        t = t1 + (t2 - t1) * (i / n)
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        c = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
        out.append(c)
    return out

def build_spline_path(positions, samples_per_seg=SAMPLES_PER_SEG):
    pts_in = [np.asarray(p, float) for p in positions]
    if len(pts_in) <= 1:
        single = np.asarray(pts_in, float) if pts_in else np.zeros((1, 3))
        return single, np.zeros(len(single)), np.zeros(len(pts_in))

    ext = ([pts_in[0] + (pts_in[0] - pts_in[1])] + pts_in +
           [pts_in[-1] + (pts_in[-1] - pts_in[-2])])

    samples = []
    wp_sample_idx = [0]
    for i in range(1, len(ext) - 2):
        samples.extend(_catmull_rom_segment(ext[i - 1], ext[i], ext[i + 1], ext[i + 2],
                                            samples_per_seg))
        wp_sample_idx.append(len(samples))
    samples.append(ext[-2])

    pts = np.asarray(samples, float)
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum_s = np.concatenate([[0.0], np.cumsum(seg_len)])
    wp_idx = np.clip(wp_sample_idx, 0, len(cum_s) - 1)
    return pts, cum_s, cum_s[wp_idx]

def path_curvature(pts, cum_s, stencil_m=CURV_STENCIL_M, horizontal_only=True):
    n = len(pts)
    curv = np.zeros(n)
    cp = pts.copy()
    if horizontal_only:
        cp[:, 2] = 0.0
    for i in range(n):
        a, b = i, i
        while a > 0 and cum_s[i] - cum_s[a] < stencil_m:
            a -= 1
        while b < n - 1 and cum_s[b] - cum_s[i] < stencil_m:
            b += 1
        if a == i or b == i:
            continue
        A, B, C = cp[a], cp[i], cp[b]
        ab = np.linalg.norm(B - A)
        bc = np.linalg.norm(C - B)
        ca = np.linalg.norm(A - C)
        denom = ab * bc * ca
        if denom < 1e-9:
            continue
        area = 0.5 * np.linalg.norm(np.cross(B - A, C - A))
        curv[i] = 4.0 * area / denom
    return curv

def speed_profile(curv, cum_s, cruise, a_lat, a_lon, end_speed=0.0):
    n = len(curv)
    v = np.minimum(cruise, np.sqrt(a_lat / np.maximum(curv, 1e-6)))
    if n == 0:
        return v
    if end_speed is not None:
        v[-1] = min(v[-1], end_speed)
    for i in range(n - 2, -1, -1):
        ds = cum_s[i + 1] - cum_s[i]
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * a_lon * ds))
    return v

def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))

class Path:
    def __init__(self, waypoints: List[np.ndarray], yaws: List[Optional[float]], loop=False, cfg=None):
        self.loop = loop
        self.cfg = cfg or {}
        
        # Build spline
        wp_pos = []
        wp_yaw = []
        for i, w in enumerate(waypoints):
            p = np.asarray(w, float)
            if wp_pos and np.linalg.norm(p - wp_pos[-1]) <= 1e-6:
                continue
            wp_pos.append(p)
            wp_yaw.append(yaws[i])
            
        if not wp_pos:
            self._pts, self._cum_s, self._wp_s = np.zeros((1, 3)), np.zeros(1), np.zeros(0)
            self._curv = np.zeros(1)
            self._vmax = np.zeros(1)
        else:
            self._pts, self._cum_s, self._wp_s = build_spline_path(wp_pos)
            self._curv = path_curvature(self._pts, self._cum_s)
            
            cruise = getattr(cfg, 'CRUISE_SPEED', 10.0) if hasattr(cfg, 'CRUISE_SPEED') else self.cfg.get('CRUISE_SPEED', 10.0)
            a_lat = getattr(cfg, 'A_LAT_MAX', 15.0) if hasattr(cfg, 'A_LAT_MAX') else self.cfg.get('A_LAT_MAX', 15.0)
            a_lon = getattr(cfg, 'A_LON_MAX', 10.0) if hasattr(cfg, 'A_LON_MAX') else self.cfg.get('A_LON_MAX', 10.0)
            finish_spd = getattr(cfg, 'FINISH_SPEED', 0.0) if hasattr(cfg, 'FINISH_SPEED') else self.cfg.get('FINISH_SPEED', 0.0)
            
            end_speed = None if self.loop else finish_spd
            self._vmax = speed_profile(self._curv, self._cum_s, cruise, a_lat, a_lon, end_speed=end_speed)
            
        self._s_end = float(self._cum_s[-1]) if len(self._cum_s) > 0 else 0.0
        self._wp_yaw = wp_yaw

    def length(self) -> float:
        return self._s_end

    def project(self, pos: np.ndarray, last_t: float) -> float:
        if len(self._pts) == 0:
            return 0.0
            
        # Find index of last_t
        idx = np.searchsorted(self._cum_s, min(max(last_t, 0.0), self._s_end))
        idx = min(idx, len(self._pts) - 1)
        
        lo = max(0, idx - BACK_WINDOW)
        hi = min(len(self._pts), idx + FWD_WINDOW + 1)
        d = np.linalg.norm(self._pts[lo:hi] - pos, axis=1)
        k = lo + int(np.argmin(d))
        return float(self._cum_s[k])

    def sample(self, t: float) -> Tuple[np.ndarray, float]:
        t = min(max(t, 0.0), self._s_end)
        j = int(np.searchsorted(self._cum_s, t))
        if j <= 0:
            return self._pts[0], self._vmax[0]
        if j >= len(self._pts):
            return self._pts[-1], self._vmax[-1]
            
        s0, s1 = self._cum_s[j - 1], self._cum_s[j]
        frac = (t - s0) / (s1 - s0) if s1 > s0 else 0.0
        p = self._pts[j - 1] + frac * (self._pts[j] - self._pts[j - 1])
        v = self._vmax[j - 1] + frac * (self._vmax[j] - self._vmax[j - 1])
        return p, v
        
    def sample_yaw(self, t: float, current_yaw: float) -> float:
        t = min(max(t, 0.0), self._s_end)
        j = int(np.searchsorted(self._wp_s, t))
        if j <= 0:
            return self._wp_yaw[0] if self._wp_yaw[0] is not None else current_yaw
        if j >= len(self._wp_s):
            return self._wp_yaw[-1] if self._wp_yaw[-1] is not None else current_yaw
            
        s0, s1 = self._wp_s[j - 1], self._wp_s[j]
        y0, y1 = self._wp_yaw[j - 1], self._wp_yaw[j]
        if y0 is None: y0 = current_yaw
        if y1 is None: y1 = y0
        
        frac = (t - s0) / (s1 - s0) if s1 > s0 else 0.0
        dy = _wrap(y1 - y0)
        return _wrap(y0 + frac * dy)

def carrot_velocity(path: Path, pos_ned: np.ndarray, vel_ned: np.ndarray, cfg, last_t: float) -> Tuple[np.ndarray, float, float, float]:
    """Pure-pursuit: speed-scaled lookahead carrot.
    Returns (vel_cmd, yaw_cmd, s_proj, lookahead).
    """
    if path.length() == 0:
        return np.zeros(3), 0.0, 0.0, 0.0
        
    s_proj = path.project(pos_ned, last_t)
    _, speed = path.sample(s_proj)
    
    speed_meas = math.hypot(vel_ned[0], vel_ned[1]) if vel_ned is not None else speed
    
    lookahead_m = getattr(cfg, 'LOOKAHEAD_M', 2.0)
    lookahead_time = getattr(cfg, 'LOOKAHEAD_TIME', 0.5)
    lookahead_max = getattr(cfg, 'LOOKAHEAD_MAX', 10.0)
    kp_vert = getattr(cfg, 'KP_VERT_PATH', 2.0)
    max_vspeed = getattr(cfg, 'MAX_VSPEED', 3.0)
    max_speed = getattr(cfg, 'MAX_SPEED', 15.0)
    
    lookahead = min(lookahead_max, max(lookahead_m, lookahead_time * speed_meas))
    s_carrot = s_proj + lookahead
    if path.loop and s_carrot > path.length():
        s_carrot -= path.length()
        
    carrot_p, _ = path.sample(s_carrot)
    to_carrot = carrot_p - pos_ned
    dist_c = float(np.linalg.norm(to_carrot))
    
    speed = min(speed, max_speed)
    
    horiz = to_carrot[:2]
    hdist = float(np.linalg.norm(horiz))
    if hdist > 1e-6:
        vn = float(horiz[0] / hdist * speed)
        ve = float(horiz[1] / hdist * speed)
    else:
        vn = ve = 0.0
        
    path_z = float(path.sample(s_proj)[0][2])
    slope_ff = float(to_carrot[2] / dist_c * speed) if dist_c > 1e-6 else 0.0
    vd = slope_ff + kp_vert * (path_z - float(pos_ned[2]))
    vd = max(-max_vspeed, min(max_vspeed, vd))
    
    vel_cmd = np.array([vn, ve, vd])
    yaw_cmd = path.sample_yaw(s_proj, 0.0)
    
    return vel_cmd, yaw_cmd, s_proj, lookahead
