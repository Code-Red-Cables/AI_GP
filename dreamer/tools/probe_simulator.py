"""Milestone 1: standalone simulator timing & interface probe.

Attaches to a running DCL FlightSim, receives camera + telemetry, sends SAFE neutral
commands, and measures the ACTUAL rates/jitter/latency (never assumes 30 Hz). Works with
no RL code. Run it BEFORE any training:

    python dreamer/tools/probe_simulator.py --seconds 30 --send-neutral

Outputs under artifacts/probe/:
    camera_timestamps.csv  telemetry_timestamps.csv  action_timestamps.csv
    rate_summary.json      probe_report.md           sample_frames/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_PKG))

from dreamer_drone.config import Config  # noqa: E402
from dreamer_drone.sim.action_sender import ActionSender  # noqa: E402
from dreamer_drone.sim.camera_io import CameraIO  # noqa: E402
from dreamer_drone.sim.mavlink_io import MavlinkIO  # noqa: E402
from dreamer_drone.sim.timing import RateMeter  # noqa: E402
from dreamer_drone.env.spaces import neutral_action  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--mavlink-port", type=int, default=14550)
    ap.add_argument("--camera-port", type=int, default=5600)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--send-neutral", action="store_true",
                    help="stream a safe hover setpoint at 50 Hz while probing")
    ap.add_argument("--out", default="artifacts/probe")
    ap.add_argument("--save-frames", type=int, default=10)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "sample_frames").mkdir(parents=True, exist_ok=True)

    cfg = Config()
    boot_ms = int(time.time() * 1000)
    io = MavlinkIO(args.host, args.mavlink_port, boot_ms)
    print(f"[probe] waiting for heartbeat on {args.host}:{args.mavlink_port} ...")
    if not io.wait_heartbeat(10.0):
        print("[probe] ERROR: no heartbeat. Is the sim running and streaming MAVLink?")
        return 1
    print(f"[probe] connected to system {io.conn.target_system}")
    io.start()
    cam = CameraIO("0.0.0.0", args.camera_port).start()
    sender = ActionSender(io, cfg.action)

    cam_meter, att_meter, imu_meter, act_meter = (RateMeter() for _ in range(4))
    cam_rows, tel_rows, act_rows = [], [], []
    saved = 0
    last_frame_id = -1
    last_att_ts = None
    last_imu_ts = None

    t0 = time.time()
    next_action = t0
    print(f"[probe] measuring for {args.seconds:.0f}s ...")
    try:
        import cv2
    except Exception:
        cv2 = None

    while time.time() - t0 < args.seconds:
        now = time.time()
        f = cam.get_latest()
        if f and f.frame_id != last_frame_id:
            last_frame_id = f.frame_id
            cam_meter.tick(f.recv_wall_s)
            latency_ms = (now - f.recv_wall_s) * 1000.0
            cam_rows.append([f.frame_id, f.sim_time_ns, f"{f.recv_wall_s:.6f}",
                             f.image_bgr.shape[1], f.image_bgr.shape[0]])
            if cv2 is not None and saved < args.save_frames:
                cv2.imwrite(str(out / "sample_frames" / f"frame_{f.frame_id:06d}.jpg"),
                            f.image_bgr)
                saved += 1

        att = io.get("attitude")
        if att and att.get("ts") != last_att_ts:
            last_att_ts = att.get("ts")
            att_meter.tick()
            tel_rows.append(["ATTITUDE", att["ts"], f"{att['yaw']:.4f}"])
        imu = io.get("highres_imu")
        if imu and imu.get("ts_us") != last_imu_ts:
            last_imu_ts = imu.get("ts_us")
            imu_meter.tick()
            tel_rows.append(["HIGHRES_IMU", imu["ts_us"], f"{imu['zacc']:.4f}"])

        if args.send_neutral and now >= next_action:
            sender.send(neutral_action())
            act_meter.tick()
            act_rows.append([f"{now:.6f}"])
            next_action = now + 0.02  # 50 Hz

        time.sleep(0.001)

    cam.stop()
    io.stop()

    # write CSVs
    _write_csv(out / "camera_timestamps.csv",
               ["frame_id", "sim_time_ns", "recv_wall_s", "width", "height"], cam_rows)
    _write_csv(out / "telemetry_timestamps.csv", ["msg", "ts", "sample"], tel_rows)
    _write_csv(out / "action_timestamps.csv", ["wall_s"], act_rows)

    def st(m):
        s = m.stats()
        return None if s is None else s.__dict__

    summary = {
        "seconds": args.seconds,
        "camera": st(cam_meter),
        "attitude": st(att_meter),
        "highres_imu": st(imu_meter),
        "actions_sent": st(act_meter),
        "frames_completed": cam.frames_completed,
        "frames_dropped": cam.frames_dropped,
        "duplicates": cam.duplicates,
        "race_status": io.get("race_status"),
        "track_gates_count": len(io.get("track_gates") or []),
        "position_available": io.get("odometry") is not None or io.get("local_position_ned") is not None,
    }
    (out / "rate_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _write_report(out / "probe_report.md", summary)
    print(f"[probe] done. Report: {out / 'probe_report.md'}")
    print(json.dumps({k: summary[k] for k in ("camera", "attitude", "highres_imu")},
                     indent=2, default=str))
    return 0


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _write_report(path: Path, s: dict) -> None:
    cam = s["camera"] or {}
    lines = [
        "# Simulator Probe Report", "",
        f"Measured over {s['seconds']:.0f}s of wall clock.", "",
        "## Measured rates", "",
        "| stream | mean Hz | mean dt (ms) | std dt (ms) | p95 dt (ms) | count |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("camera", "attitude", "highres_imu", "actions_sent"):
        d = s.get(name) or {}
        lines.append(f"| {name} | {d.get('mean_hz', 0):.2f} | {d.get('mean_dt_ms', 0):.2f} "
                     f"| {d.get('std_dt_ms', 0):.2f} | {d.get('p95_dt_ms', 0):.2f} "
                     f"| {d.get('count', 0)} |")
    lines += [
        "", "## Camera health", "",
        f"- frames completed: {s['frames_completed']}",
        f"- incomplete frames dropped: {s['frames_dropped']}",
        f"- duplicate frame ids: {s['duplicates']}",
        "", "## Privileged / race state", "",
        f"- race_status present: {s['race_status'] is not None}",
        f"- track gates received: {s['track_gates_count']}",
        f"- position telemetry available (VQ2 expects False): {s['position_available']}",
        "", "## Notes",
        "- Compare camera mean Hz against the spec's claimed 30 Hz.",
        "- If std dt is large, the decision rate / watchdogs must tolerate jitter.",
        "- `position_available=True` would enable privileged dense-progress reward.",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
