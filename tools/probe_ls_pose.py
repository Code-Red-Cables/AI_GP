"""Ground-truth check on the LS gate pose, especially its vertical channel.

Places the camera at a known offset from a gate, projects the eight keypoints
through the real camera model, and asks ``solve_keypoints_ls`` to recover the
offset. Sweeping body pitch answers the question a flight cannot: does the
solved vertical stay honest while the drone leans forward to drive, or does the
lean leak into it?

    python tools/probe_ls_pose.py
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import camera_model as cm  # noqa: E402
from vision.gate_ls_pose import solve_keypoints_ls  # noqa: E402
from vision.yolo_pnp import KEYPOINT_OBJECT_POINTS  # noqa: E402

# The gate frame is x=right, y=down, z=through-the-hoop. A gate at yaw 0 faces
# north, so mapping it into NED stands it upright:
#   gate x (right)   -> NED y (east)
#   gate y (down)    -> NED z (down)
#   gate z (through) -> NED x (north)
# Getting this wrong lays the gate flat on the ground, which is a degenerate
# view from any sane camera pose and makes the solver look broken when it is not.
R_NED_GATE = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])


def gate_points_ned() -> np.ndarray:
    return (R_NED_GATE @ np.asarray(KEYPOINT_OBJECT_POINTS, float).T).T


def project_gate(cam_ned, roll, pitch, yaw):
    """Pixel coordinates of the eight gate keypoints, or None if behind."""
    R_wb = cm.rot_world_body(roll, pitch, yaw)
    pts = []
    for p in gate_points_ned():
        rel_body = R_wb.T @ (p - np.asarray(cam_ned, dtype=float))
        rel_cam = cm.body_to_cam(rel_body)
        if rel_cam[2] <= 0.05:
            return None
        pts.append(cm.project(rel_cam))
    return np.asarray(pts, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--range', type=float, default=10.0,
                    help='metres in front of the gate')
    ap.add_argument('--below', type=float, default=1.0,
                    help='metres the camera sits below gate centre (NED +z)')
    ap.add_argument('--lateral', type=float, default=0.0,
                    help='metres the camera sits right of the centreline')
    args = ap.parse_args()

    cam_ned = np.array([-abs(args.range), args.lateral, args.below])
    truth_range = float(np.linalg.norm(cam_ned))
    print(f'truth: range={truth_range:.2f} m  below={args.below:+.2f} m  '
          f'lateral={args.lateral:+.2f} m')
    print(f'{"pitch":>7s} {"vertical":>9s} {"lateral":>8s} {"range":>7s} '
          f'{"resid":>6s}   note')

    for pitch_deg in (0.0, 5.0, 10.0, 20.0, 30.0, 35.0):
        pitch = math.radians(pitch_deg)
        pts = project_gate(cam_ned, 0.0, pitch, 0.0)
        if pts is None:
            print(f'{pitch_deg:7.1f} {"":>9s} {"":>8s} {"":>7s} {"":>6s}   '
                  'gate out of view')
            continue
        pose = solve_keypoints_ls(pts, [1.0] * 8, roll=0.0, pitch=pitch, yaw=0.0)
        if pose is None:
            print(f'{pitch_deg:7.1f} {"":>9s} {"":>8s} {"":>7s} {"":>6s}   '
                  'solve failed')
            continue
        err = pose.vertical_m - args.below
        note = 'ok' if abs(err) < 0.25 else f'vertical off by {err:+.2f} m'
        print(f'{pitch_deg:7.1f} {pose.vertical_m:+9.3f} {pose.lateral_m:+8.3f} '
              f'{pose.range_m:7.2f} {pose.residual_m:6.3f}   {note}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
