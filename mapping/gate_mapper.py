"""GateMapper — fuse per-frame gate observations into persistent world-frame estimates.

Implements docs/VISION_REBOOT_PLAN.md §5.3 (see docs/GAP_ANALYSIS.md G-4/G-5 for why the
previous 64-line EMA version was insufficient). Pure module: no threads, no cv2, no
shared_data access — inputs are observation dicts/objects + the drone position, outputs
are GateEstimates and position-fix deltas. Deterministic and unit-testable offline.

Filter design (kept deliberately simpler than a full EKF):

* Per-gate, per-axis scalar Kalman on position. Measurement variance
  ``R = (PNP_SIGMA0_M + PNP_SIGMA_K * range)²``, multiplied by SIZE_METHOD_VAR_MULT for
  size-method observations, divided by (bounded) detection confidence, and the D
  (vertical) axis inflated by ``max(1, (range / FAR_VERT_RANGE_M)²)`` — that last term
  encodes hard-won fact 6 (the +20°-up camera makes far elevation untrustworthy) as
  measurement noise instead of a special-cased hack.
* Association by normalized squared distance ``Σ (z-x)²/(P+R)`` against every mapped
  gate; best match under GATE_ASSOC_CHI2 wins, else a new candidate spawns.
* A candidate becomes ``confirmed`` after MIN_OBS_CONFIRM observations inside
  CONFIRM_WINDOW_S; unconfirmed candidates unseen for CANDIDATE_PRUNE_S are pruned —
  together these kill the garbage-burst phantoms (fact 10).
* Normal: flipped into the hemisphere facing the drone on first sight, then hemisphere-
  locked and averaged (re-normalized). Size-method observations carry no real normal and
  never touch it.
* ``update()`` returns position-fix deltas (plan §9b / Phase 2b): an observation that
  associates to a CONFIRMED gate implies the drone is displaced by
  ``delta = gate.pos − obs.gate_ned`` (both embed the same drone-position term, so the
  gate's filtered position corrects the drone's). The state estimator applies these.
"""

import json
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Defaults double as documentation; a cfg object (config.py module) overrides any of them.
DEFAULTS = {
    'GATE_ASSOC_CHI2': 9.0,        # association gate on Σ (z-x)²/(P+R)  (~3σ per axis)
    'MIN_OBS_CONFIRM': 3,          # observations needed inside the window to confirm
    'CONFIRM_WINDOW_S': 2.0,       # window (s) for the confirmation count
    'CANDIDATE_PRUNE_S': 3.0,      # unconfirmed candidate unseen this long -> dropped
    'PNP_SIGMA0_M': 0.3,           # base measurement σ (m) at zero range, PnP method
    'PNP_SIGMA_K': 0.05,           # σ growth per metre of range
    'SIZE_METHOD_VAR_MULT': 4.0,   # variance multiplier for the size-method fallback
    'FAR_VERT_RANGE_M': 12.0,      # D-axis variance inflation knee (fact 6)
    'VIS_CONF_MIN': 0.10,          # discard observations below this confidence
    'VIS_MAX_RANGE_M': 40.0,       # discard observations beyond this range
    'POS_FIX_MAX_RANGE_M': 20.0,   # only near observations of confirmed gates yield fixes
}


def _cfg_get(cfg, key):
    if cfg is not None and hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict) and key in cfg:
        return cfg[key]
    return DEFAULTS[key]


def _obs_get(obs, key, default=None):
    """Read a field from a GateObservation dataclass OR a plain dict."""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _as_vec(v):
    """None-safe conversion to a finite (3,) float array, else None."""
    if v is None:
        return None
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.shape[0] != 3 or not np.all(np.isfinite(a)):
        return None
    return a


@dataclass
class GateEstimate:
    gate_id: int
    pos: np.ndarray                  # (3,) filtered NED center
    normal: np.ndarray               # (3,) unit normal, hemisphere locked at first sight
    pos_var: np.ndarray              # (3,) per-axis variance (m²)
    n_obs: int = 1
    first_seen_ns: int = 0
    last_seen_ns: int = 0
    confirmed: bool = False
    passed: bool = False
    obs_times_ns: list = field(default_factory=list)   # recent obs times (confirm window)

    @property
    def yaw(self) -> float:
        """Heading of the gate normal (rad, NED atan2(E, N)) — approach axis."""
        return float(math.atan2(self.normal[1], self.normal[0]))

    def to_dict(self) -> dict:
        return {
            'gate_id': self.gate_id,
            'pos': [float(v) for v in self.pos],
            'normal': [float(v) for v in self.normal],
            'pos_var': [float(v) for v in self.pos_var],
            'n_obs': self.n_obs,
            'first_seen_ns': self.first_seen_ns,
            'last_seen_ns': self.last_seen_ns,
            'confirmed': self.confirmed,
            'passed': self.passed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'GateEstimate':
        return cls(
            gate_id=int(d['gate_id']),
            pos=np.asarray(d['pos'], dtype=float),
            normal=np.asarray(d['normal'], dtype=float),
            pos_var=np.asarray(d['pos_var'], dtype=float),
            n_obs=int(d.get('n_obs', 1)),
            first_seen_ns=int(d.get('first_seen_ns', 0)),
            last_seen_ns=int(d.get('last_seen_ns', 0)),
            confirmed=bool(d.get('confirmed', True)),
            passed=bool(d.get('passed', False)),
        )


class GateMapper:

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.gates: List[GateEstimate] = []
        self._next_id = 1

    # -- measurement model ---------------------------------------------------------

    def _meas_var(self, range_m: float, method: str, confidence: float) -> np.ndarray:
        sigma = _cfg_get(self.cfg, 'PNP_SIGMA0_M') + _cfg_get(self.cfg, 'PNP_SIGMA_K') * range_m
        var = sigma * sigma
        if method != 'pnp':
            var *= _cfg_get(self.cfg, 'SIZE_METHOD_VAR_MULT')
        # Low confidence -> noisier measurement (bounded so conf can't zero the update).
        var /= min(max(confidence, 0.25), 1.0)
        r = np.array([var, var, var])
        knee = _cfg_get(self.cfg, 'FAR_VERT_RANGE_M')
        r[2] *= max(1.0, (range_m / knee) ** 2)   # far elevation is untrustworthy (fact 6)
        return r

    # -- update --------------------------------------------------------------------

    def update(self, observations, drone_pos_ned=None, now_ns: Optional[int] = None) -> List[dict]:
        """Integrate one frame's observations. Returns position-fix deltas (may be []).

        Each fix: {'delta': [dn, de, dd], 'var': [...], 'ts_ns': int, 'gate_id': int}
        meaning "the drone's position estimate is off by -delta", i.e. the consumer
        should nudge its position BY +delta (see plan §9b).
        """
        fixes: List[dict] = []
        drone_pos = _as_vec(drone_pos_ned)
        latest_ns = now_ns or 0

        for obs in observations or []:
            z = _as_vec(_obs_get(obs, 'gate_ned'))
            if z is None:
                continue                      # no world projection (no attitude yet)
            conf = float(_obs_get(obs, 'confidence', 0.0) or 0.0)
            if conf < _cfg_get(self.cfg, 'VIS_CONF_MIN'):
                continue
            range_m = float(_obs_get(obs, 'range_m', 0.0) or 0.0)
            if not (0.0 < range_m <= _cfg_get(self.cfg, 'VIS_MAX_RANGE_M')):
                continue
            method = _obs_get(obs, 'method', 'size') or 'size'
            ts_ns = int(_obs_get(obs, 'ts_ns', 0) or 0)
            latest_ns = max(latest_ns, ts_ns)

            r_var = self._meas_var(range_m, method, conf)

            # -- associate: best normalized distance under the chi2 gate --
            best, best_d2 = None, _cfg_get(self.cfg, 'GATE_ASSOC_CHI2')
            for g in self.gates:
                d2 = float(np.sum((z - g.pos) ** 2 / (g.pos_var + r_var)))
                if d2 < best_d2:
                    best, best_d2 = g, d2

            normal = self._observed_normal(obs, z, drone_pos)

            if best is None:
                self.gates.append(GateEstimate(
                    gate_id=self._next_id,
                    pos=z.copy(),
                    normal=normal if normal is not None else np.zeros(3),
                    pos_var=r_var.copy(),
                    n_obs=1,
                    first_seen_ns=ts_ns,
                    last_seen_ns=ts_ns,
                    confirmed=(_cfg_get(self.cfg, 'MIN_OBS_CONFIRM') <= 1),
                    obs_times_ns=[ts_ns],
                ))
                self._next_id += 1
                continue

            # Position fix BEFORE the measurement moves the gate (don't feed z back).
            if (best.confirmed and method == 'pnp'
                    and range_m <= _cfg_get(self.cfg, 'POS_FIX_MAX_RANGE_M')):
                fixes.append({
                    'delta': [float(v) for v in (best.pos - z)],
                    'var': [float(v) for v in (best.pos_var + r_var)],
                    'ts_ns': ts_ns,
                    'gate_id': best.gate_id,
                })

            # -- scalar Kalman per axis --
            k = best.pos_var / (best.pos_var + r_var)
            best.pos = best.pos + k * (z - best.pos)
            best.pos_var = (1.0 - k) * best.pos_var
            best.n_obs += 1
            best.last_seen_ns = max(best.last_seen_ns, ts_ns)

            # -- confirmation: MIN_OBS_CONFIRM observations inside the window --
            if not best.confirmed:
                window_ns = int(_cfg_get(self.cfg, 'CONFIRM_WINDOW_S') * 1e9)
                best.obs_times_ns.append(ts_ns)
                best.obs_times_ns = [t for t in best.obs_times_ns if ts_ns - t <= window_ns]
                if len(best.obs_times_ns) >= _cfg_get(self.cfg, 'MIN_OBS_CONFIRM'):
                    best.confirmed = True
                    best.obs_times_ns = []

            # -- normal: hemisphere-locked running mean (PnP observations only) --
            if normal is not None:
                stored = best.normal
                if float(np.linalg.norm(stored)) < 0.5:
                    best.normal = normal
                else:
                    if float(np.dot(normal, stored)) < 0.0:
                        normal = -normal      # keep the first-seen hemisphere
                    w = min(best.n_obs, 20)   # bounded memory so the normal can still adapt
                    merged = stored * w + normal
                    n = float(np.linalg.norm(merged))
                    if n > 1e-9:
                        best.normal = merged / n

        self._prune(latest_ns)
        return fixes

    def _observed_normal(self, obs, gate_ned: np.ndarray, drone_pos) -> Optional[np.ndarray]:
        """Unit normal in NED, flipped to face the drone's side; None for size method."""
        if (_obs_get(obs, 'method', 'size') or 'size') != 'pnp':
            return None
        n = _as_vec(_obs_get(obs, 'normal_ned'))
        if n is None:
            return None
        norm = float(np.linalg.norm(n))
        if norm < 0.5:
            return None                       # zero placeholder — no real normal
        n = n / norm
        # Primary flip: body-frame test (drone at origin: gate_body points drone->gate,
        # a drone-facing normal satisfies dot(normal_body, gate_body) < 0).
        nb = _as_vec(_obs_get(obs, 'normal_body'))
        gb = _as_vec(_obs_get(obs, 'gate_body'))
        if nb is not None and gb is not None and float(np.linalg.norm(nb)) > 0.5:
            if float(np.dot(nb, gb)) > 0.0:
                n = -n
        elif drone_pos is not None:
            if float(np.dot(n, drone_pos - gate_ned)) < 0.0:
                n = -n
        return n

    def _prune(self, now_ns: int):
        if now_ns <= 0:
            return
        ttl_ns = int(_cfg_get(self.cfg, 'CANDIDATE_PRUNE_S') * 1e9)
        self.gates = [g for g in self.gates
                      if g.confirmed or (now_ns - g.last_seen_ns) <= ttl_ns]

    # -- queries ---------------------------------------------------------------------

    def course(self) -> List[GateEstimate]:
        """Confirmed gates, in confirmation (course-encounter) order."""
        return [g for g in self.gates if g.confirmed]

    def gate(self, gate_id: int) -> Optional[GateEstimate]:
        for g in self.gates:
            if g.gate_id == gate_id:
                return g
        return None

    def next_unpassed(self, ref_pos=None, ref_normal=None) -> Optional[GateEstimate]:
        """Next confirmed unpassed gate.

        With a reference (typically the just-passed gate's exit point + normal), prefer
        the nearest unpassed gate that lies in the forward (exit) hemisphere — this makes
        gate ordering robust to the mapper confirming gates out of course order (G-8).
        Falls back to nearest overall, then to confirmation order.
        """
        cands = [g for g in self.gates if g.confirmed and not g.passed]
        if not cands:
            return None
        ref_pos = _as_vec(ref_pos)
        if ref_pos is None:
            return cands[0]
        ref_normal = _as_vec(ref_normal)
        ahead = cands
        if ref_normal is not None and float(np.linalg.norm(ref_normal)) > 1e-6:
            ahead = [g for g in cands
                     if float(np.dot(g.pos - ref_pos, ref_normal)) > -1.0] or cands
        return min(ahead, key=lambda g: float(np.linalg.norm(g.pos - ref_pos)))

    def mark_passed(self, gate_id: int):
        g = self.gate(gate_id)
        if g is not None:
            g.passed = True

    # -- persistence -------------------------------------------------------------------

    def to_json(self) -> dict:
        return {'version': 1,
                'gates': [g.to_dict() for g in self.gates if g.confirmed]}

    @classmethod
    def from_json(cls, d: dict, cfg=None) -> 'GateMapper':
        m = cls(cfg)
        for gd in d.get('gates', []):
            m.gates.append(GateEstimate.from_dict(gd))
        m._next_id = max((g.gate_id for g in m.gates), default=0) + 1
        return m

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_json(), f, indent=1)

    @classmethod
    def load(cls, path: str, cfg=None) -> 'GateMapper':
        with open(path) as f:
            return cls.from_json(json.load(f), cfg)

    # -- publishing helper (plain types only — logger/json safe) ------------------------

    def snapshot(self) -> dict:
        return {'gates': [g.to_dict() for g in self.gates],
                'n_confirmed': len(self.course())}
