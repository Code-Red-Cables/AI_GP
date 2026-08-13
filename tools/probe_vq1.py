"""Characterize what the VQ1 simulator actually publishes.

Phase 1 of the HG-DAgger plan. Answers three questions that everything
downstream depends on:

  1. Which MAVLink messages arrive, and at what rate?
  2. Does ODOMETRY carry usable ground truth, and in what units?
  3. Is TRACK_DATA live (gate poses populated) or nulled?

Question 3 decides whether the gate map is free or has to be flown. The
official VQ1 client (PyAIPilotExample-v4) ships handlers for ODOMETRY,
LOCAL_POSITION_NED and TRACK_DATA but annotates each as "disabled" / "will be
nulled" -- a handler existing proves nothing, so this measures instead.

Wire formats below are copied from that official client so the decode matches
the simulator exactly:

  ENCAPSULATED_DATA payload[0]      = 1 -> race status, 2 -> track info
  race status                       "<BQqqIq"
  track chunk header                "<BH" then payload[3:], keyed by msg.seqnr
  DATA_TRANSMISSION_HANDSHAKE       msg.width = transfer id, msg.packets = count
  track payload                     "<H" num_gates, then "<Hfffffffff" per gate

Usage (with the VQ1 sim running and a race started):

    .\\winvenv\\Scripts\\python.exe tools\\probe_vq1.py [--seconds 20]
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from collections import defaultdict
from pathlib import Path

from pymavlink import mavutil

ROOT = Path(__file__).resolve().parent.parent

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID = 2

GATE_STRUCT = "<Hfffffffff"
GATE_BYTES = 38


def _finite(*values) -> bool:
    for v in values:
        if v is None:
            return False
        try:
            f = float(v)
        except (TypeError, ValueError):
            return False
        if f != f:  # NaN
            return False
    return True


def decode_track(payload: bytes) -> list[dict]:
    """Decode an assembled track-info payload into per-gate records."""
    gates: list[dict] = []
    if len(payload) < 2:
        return gates
    (num_gates,) = struct.unpack_from("<H", payload)
    body = payload[2:]
    for _ in range(num_gates):
        if len(body) < GATE_BYTES:
            break
        (gid, px, py, pz, qw, qx, qy, qz, width, height) = struct.unpack_from(
            GATE_STRUCT, body
        )
        gates.append({
            "gate_id": int(gid),
            "position_ned": [px, py, pz],
            "orientation_ned_wxyz": [qw, qx, qy, qz],
            "width": width,
            "height": height,
        })
        body = body[GATE_BYTES:]
    return gates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--conn", default="udpin:127.0.0.1:14550")
    ap.add_argument(
        "--out", type=Path, default=ROOT / "artifacts" / "vq1_probe.json"
    )
    args = ap.parse_args()

    print(f"listening on {args.conn} for {args.seconds:.0f}s ...", flush=True)
    conn = mavutil.mavlink_connection(args.conn)

    counts: dict[str, int] = defaultdict(int)
    odom_samples: list[dict] = []
    attitude_sample: dict | None = None
    imu_sample: dict | None = None
    local_pos_sample: dict | None = None
    race_samples: list[dict] = []
    track_gates: list[dict] = []
    track_chunks: dict[int, dict[int, bytes]] = {}
    expected_chunks: dict[int, int] = {}

    t0 = time.time()
    while time.time() - t0 < args.seconds:
        msg = conn.recv_match(blocking=True, timeout=0.5)
        if msg is None:
            continue
        mtype = msg.get_type()
        if mtype == "BAD_DATA":
            counts["BAD_DATA"] += 1
            continue
        counts[mtype] += 1

        if mtype == "ODOMETRY" and len(odom_samples) < 5:
            q = list(getattr(msg, "q", []) or [])
            odom_samples.append({
                "x": msg.x, "y": msg.y, "z": msg.z,
                "vx": msg.vx, "vy": msg.vy, "vz": msg.vz,
                "rollspeed": getattr(msg, "rollspeed", None),
                "pitchspeed": getattr(msg, "pitchspeed", None),
                "yawspeed": getattr(msg, "yawspeed", None),
                "q_wxyz": [float(v) for v in q] if len(q) == 4 else None,
                "time_usec": getattr(msg, "time_usec", None),
                "reset_counter": getattr(msg, "reset_counter", None),
            })
        elif mtype == "ATTITUDE" and attitude_sample is None:
            attitude_sample = {
                "roll": msg.roll, "pitch": msg.pitch, "yaw": msg.yaw,
                "rollspeed": msg.rollspeed,
                "pitchspeed": msg.pitchspeed,
                "yawspeed": msg.yawspeed,
            }
        elif mtype == "HIGHRES_IMU" and imu_sample is None:
            imu_sample = {
                "xacc": msg.xacc, "yacc": msg.yacc, "zacc": msg.zacc,
                "xgyro": msg.xgyro, "ygyro": msg.ygyro, "zgyro": msg.zgyro,
            }
        elif mtype == "LOCAL_POSITION_NED" and local_pos_sample is None:
            local_pos_sample = {
                "x": msg.x, "y": msg.y, "z": msg.z,
                "vx": msg.vx, "vy": msg.vy, "vz": msg.vz,
            }
        elif mtype == "DATA_TRANSMISSION_HANDSHAKE":
            tid = int(getattr(msg, "width", 0))
            track_chunks[tid] = {}
            expected_chunks[tid] = int(getattr(msg, "packets", 0))
        elif mtype == "ENCAPSULATED_DATA":
            raw = bytes(msg.data)
            if not raw:
                continue
            dtype = raw[0]
            if dtype == ENCAPSULATED_RACE_STATUS_MSG_ID and len(race_samples) < 3:
                try:
                    fields = struct.unpack_from("<BQqqIq", raw)
                    race_samples.append({
                        "sim_boot_time_ms": fields[1],
                        "race_start_boot_time_ms": fields[2],
                        "race_finish_time_ns": fields[3],
                        "active_gate_index": fields[4],
                        "last_gate_race_time": fields[5],
                    })
                except struct.error:
                    pass
            elif dtype == ENCAPSULATED_TRACK_INFO_MSG_ID:
                try:
                    _dt, tid = struct.unpack_from("<BH", raw)
                except struct.error:
                    continue
                if tid not in expected_chunks:
                    continue
                track_chunks.setdefault(tid, {})[int(msg.seqnr)] = raw[3:]
                if len(track_chunks[tid]) == expected_chunks[tid]:
                    full = b"".join(
                        track_chunks[tid][i] for i in sorted(track_chunks[tid])
                    )
                    track_gates = decode_track(full)
                    track_chunks.pop(tid, None)
                    expected_chunks.pop(tid, None)

    span = max(1e-6, time.time() - t0)

    # --- verdicts -------------------------------------------------------
    odom_live = bool(odom_samples) and any(
        _finite(s["x"], s["y"], s["z"]) and any(
            abs(float(s[k])) > 1e-9 for k in ("x", "y", "z")
        )
        for s in odom_samples
    )
    track_live = bool(track_gates) and any(
        _finite(*g["position_ned"]) and any(
            abs(float(v)) > 1e-9 for v in g["position_ned"]
        )
        for g in track_gates
    )

    report = {
        "conn": args.conn,
        "seconds": round(span, 2),
        "rates_hz": {k: round(v / span, 1) for k, v in sorted(counts.items())},
        "counts": dict(sorted(counts.items())),
        "odometry_live": odom_live,
        "odometry_samples": odom_samples,
        "attitude_sample": attitude_sample,
        "highres_imu_sample": imu_sample,
        "local_position_ned_sample": local_pos_sample,
        "race_status_samples": race_samples,
        "track_data_live": track_live,
        "track_num_gates": len(track_gates),
        "track_gates": track_gates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    # --- human summary --------------------------------------------------
    print(f"\n=== message rates over {span:.1f}s ===")
    if not counts:
        print("  NO PACKETS — is the sim running with a race started?")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<32} {v:>6}  ({v / span:5.1f} Hz)")

    print("\n=== ground truth ===")
    print(f"  ODOMETRY live:   {odom_live}")
    if odom_samples:
        s = odom_samples[0]
        print(f"    pos  = ({s['x']:.3f}, {s['y']:.3f}, {s['z']:.3f}) m")
        print(f"    vel  = ({s['vx']:.3f}, {s['vy']:.3f}, {s['vz']:.3f}) m/s")
        print(f"    q    = {s['q_wxyz']}")
    print(f"  LOCAL_POSITION_NED seen: {local_pos_sample is not None}")

    print(f"\n  TRACK_DATA live: {track_live}  ({len(track_gates)} gates)")
    for g in track_gates[:3]:
        p = g["position_ned"]
        print(f"    gate {g['gate_id']:>2}  ned=({p[0]:.2f}, {p[1]:.2f}, "
              f"{p[2]:.2f})  {g['width']:.2f}x{g['height']:.2f} m")
    if track_gates and len(track_gates) > 3:
        print(f"    ... {len(track_gates) - 3} more")

    print("\n=== what this means for the plan ===")
    # Spec section 4.3 only guarantees HEARTBEAT, ATTITUDE, HIGHRES_IMU and
    # TIMESYNC. ODOMETRY was a bonus on the 3391 build and is not something to
    # design around; ATTITUDE is, because it is the only drift-free attitude
    # source the policy can use.
    if attitude_sample is not None:
        print("  ATTITUDE present — the policy's attitude channel is sound.")
    else:
        print("  ATTITUDE MISSING. The observation then falls back to the")
        print("  controller AHRS, and if that is unavailable to the EKF belief,")
        print("  which drifts tens of degrees per minute. Check this first.")
    if imu_sample is not None:
        print("  HIGHRES_IMU present — body rates available.")
    else:
        print("  HIGHRES_IMU MISSING — no body-rate channel.")
    if odom_live:
        print("  ODOMETRY present (bonus): the ground-truth bearing check in")
        print("  tools/eval_observation.py will run as check 0.")
    else:
        print("  ODOMETRY absent — expected on current builds. The observation")
        print("  gate runs ground-truth free (rigidity / identity / coupling);")
        print("  nothing in the policy needed it. Run:")
        print("    python tools/eval_observation.py --telem logs/telem_*.csv")
    if track_live:
        print("  Gate map is FREE — decode TRACK_DATA into the course map.")
    else:
        print("  No gate map. Only the optional ODOMETRY check wanted one.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
