"""Control-response & reset probe (the second half of Milestone 1).

Unlike `probe_simulator.py` (which only sends neutral), this arms the drone and sends
**single-axis physical pulses**, measuring the gyro/accel response so you can read off:

  1. `cmd 31000` reset semantics — does active_gate reset to 0, does race_start change,
     and how long until a fresh frame arrives.
  2. Whether ACRO body-rate control responds immediately post-arm.
  3. The SIGN of each axis (roll/pitch/yaw rate + thrust) — feeds config.action.rate_sign_*.

Commands are sent RAW (bypassing the action-sign config) so the true sim convention is
discovered, not assumed. Pulses are short and modest, returning to neutral between them,
but the drone WILL move — run it from a safe reset state.

    python dreamer/tools/probe_control.py --host 127.0.0.1

Outputs: artifacts/probe/control_log.csv, artifacts/probe/control_report.md
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_PKG))

from dreamer_drone.config import Config  # noqa: E402
from dreamer_drone.sim.mavlink_io import MavlinkIO  # noqa: E402
from dreamer_drone.sim.camera_io import CameraIO  # noqa: E402

RATE = 1.5          # rad/s pulse magnitude for roll/pitch/yaw tests
PULSE_S = 0.6       # duration of each pulse
NEUTRAL_S = 0.8     # settle time between pulses
SEND_HZ = 50.0


def _stream(io, cam, thrust, roll, pitch, yaw, seconds, hover, rows, label):
    """Send a fixed physical command for `seconds`, sampling gyro/accel; return means."""
    gx, gy, gz, az = [], [], [], []
    t_end = time.time() + seconds
    while time.time() < t_end:
        io.send_attitude_target(thrust, roll, pitch, yaw)
        imu = io.get("highres_imu") or {}
        gx.append(imu.get("xgyro", 0.0)); gy.append(imu.get("ygyro", 0.0))
        gz.append(imu.get("zgyro", 0.0)); az.append(imu.get("zacc", 0.0))
        rows.append([f"{time.time():.4f}", label, thrust, roll, pitch, yaw,
                     imu.get("xgyro", 0.0), imu.get("ygyro", 0.0),
                     imu.get("zgyro", 0.0), imu.get("zacc", 0.0)])
        time.sleep(1.0 / SEND_HZ)
    mean = lambda v: statistics.fmean(v) if v else 0.0
    return {"gx": mean(gx), "gy": mean(gy), "gz": mean(gz), "az": mean(az)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--mavlink-port", type=int, default=14550)
    ap.add_argument("--camera-port", type=int, default=5600)
    ap.add_argument("--out", default="artifacts/probe")
    ap.add_argument("--no-arm", action="store_true")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    hover = cfg.action.hover_thrust
    boot_ms = int(time.time() * 1000)

    io = MavlinkIO(args.host, args.mavlink_port, boot_ms)
    print(f"[ctrl] waiting for heartbeat on {args.host}:{args.mavlink_port} ...")
    if not io.wait_heartbeat(10.0):
        print("[ctrl] ERROR: no heartbeat.")
        return 1
    io.start()
    cam = CameraIO("0.0.0.0", args.camera_port).start()
    time.sleep(1.0)

    # --- reset semantics ---
    rs_before = io.get("race_status")
    print(f"[ctrl] race_status before reset: {rs_before}")
    io.send_reset()
    t_reset = time.time()
    last_frame = cam.get_latest()
    last_id = last_frame.frame_id if last_frame else -1
    first_fresh_dt = None
    while time.time() - t_reset < 5.0:
        f = cam.get_latest()
        if f and f.frame_id != last_id:
            first_fresh_dt = time.time() - t_reset
            break
        time.sleep(0.005)
    time.sleep(1.5)
    rs_after = io.get("race_status")
    print(f"[ctrl] race_status after reset:  {rs_after}")

    rows: list[list] = []
    responses: dict[str, dict] = {}
    if not args.no_arm:
        io.arm()
        # prime neutral stream so ACRO stays live
        _stream(io, cam, hover, 0, 0, 0, 1.0, hover, rows, "arm_neutral")
        base = _stream(io, cam, hover, 0, 0, 0, NEUTRAL_S, hover, rows, "baseline")

        tests = [
            ("roll_pos",  hover, +RATE, 0, 0),
            ("roll_neg",  hover, -RATE, 0, 0),
            ("pitch_pos", hover, 0, +RATE, 0),
            ("pitch_neg", hover, 0, -RATE, 0),
            ("yaw_pos",   hover, 0, 0, +RATE),
            ("yaw_neg",   hover, 0, 0, -RATE),
            ("thrust_up", hover + 0.3, 0, 0, 0),
            ("thrust_dn", max(0.05, hover - 0.2), 0, 0, 0),
        ]
        for label, thr, r, p, y in tests:
            responses[label] = _stream(io, cam, thr, r, p, y, PULSE_S, hover, rows, label)
            _stream(io, cam, hover, 0, 0, 0, NEUTRAL_S, hover, rows, "neutral")
        # settle neutral
        _stream(io, cam, hover, 0, 0, 0, 1.0, hover, rows, "end_neutral")

    cam.stop(); io.stop()

    # --- write log ---
    with open(out / "control_log.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["wall_s", "phase", "cmd_thrust", "cmd_roll", "cmd_pitch", "cmd_yaw",
                    "gyro_x", "gyro_y", "gyro_z", "acc_z"])
        w.writerows(rows)

    _report(out / "control_report.md", rs_before, rs_after, first_fresh_dt,
            base if not args.no_arm else None, responses)
    print(f"[ctrl] done. Report: {out / 'control_report.md'}")
    return 0


def _sign(v: float, dead: float) -> str:
    if abs(v) < dead:
        return "0 (no response)"
    return "+1" if v > 0 else "-1"


def _report(path, rs_before, rs_after, first_fresh_dt, base, resp):
    lines = ["# Control & Reset Probe Report", ""]
    lines += ["## Reset (cmd 31000)", ""]

    def gate(rs):
        return None if not rs else rs.get("active_gate")

    def start(rs):
        return None if not rs else rs.get("race_start_ms")

    lines += [
        f"- active_gate: before={gate(rs_before)} -> after={gate(rs_after)} "
        f"({'RESET to 0' if gate(rs_after) == 0 and gate(rs_before) not in (0, None) else 'check'})",
        f"- race_start_ms: before={start(rs_before)} -> after={start(rs_after)} "
        f"({'changed' if start(rs_before) != start(rs_after) else 'unchanged'})",
        f"- time to first fresh frame after reset: "
        f"{f'{first_fresh_dt*1000:.0f} ms' if first_fresh_dt is not None else 'no fresh frame in 5 s (!)'}",
        "",
    ]

    if base is None:
        lines += ["## Control", "", "_(--no-arm: control test skipped)_"]
    else:
        d = 0.05  # rad/s deadband for "responded"
        lines += ["## Control response (raw physical commands)", "",
                  f"baseline gyro (rad/s): x={base['gx']:+.3f} y={base['gy']:+.3f} z={base['gz']:+.3f}",
                  "", "| pulse | cmd | measured axis (mean, baseline-subtracted) | inferred sign |",
                  "|---|---|---|---|"]

        def delta(label, axis):
            return resp[label][axis] - base[{"gx": "gx", "gy": "gy", "gz": "gz"}[axis]]

        for label, axis, name in [("roll_pos", "gx", "gyro_x"), ("pitch_pos", "gy", "gyro_y"),
                                  ("yaw_pos", "gz", "gyro_z")]:
            dv = delta(label, axis)
            lines.append(f"| {label} (+{RATE}) | +rate | Δ{name}={dv:+.3f} | {_sign(dv, d)} |")
        # thrust authority via acc_z
        if "thrust_up" in resp:
            daz = resp["thrust_up"]["az"] - base["az"]
            lines.append(f"| thrust_up (+0.3) | +thrust | Δacc_z={daz:+.3f} | "
                         f"{'responds' if abs(daz) > 0.2 else 'NO response (check thrust)'} |")
        lines += ["", "## Recommended config.action signs",
                  "For each axis: if commanding +rate produced +gyro, keep sign +1; if it",
                  "produced -gyro, set that rate_sign to -1 (the sim inverts that axis).",
                  "",
                  f"- rate_sign_roll:  {_sign(delta('roll_pos', 'gx'), 0.05)}",
                  f"- rate_sign_pitch: {_sign(delta('pitch_pos', 'gy'), 0.05)}   "
                  f"(config default is -1)",
                  f"- rate_sign_yaw:   {_sign(delta('yaw_pos', 'gz'), 0.05)}"]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
