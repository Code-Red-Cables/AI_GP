"""Autonomous policy flyer — no gamepad, no H/T intervention.

The policy owns roll / pitch / yaw-rate / throttle from arm until you quit.
Esc in the console or Ctrl+C disarms. That is the only human input.

    .\\winvenv\\Scripts\\python.exe tools\\run_policy.py
    .\\winvenv\\Scripts\\python.exe tools\\run_policy.py --panel
    .\\winvenv\\Scripts\\python.exe tools\\run_policy.py --weights models\\policy_seed_17.pt --yolo models\\gate_pose_v5.pt
    .\\winvenv\\Scripts\\python.exe tools\\tune_flight.py policy --panel

Start a race in the FlightSim window and stay on the pad facing gate 1
before the client arms.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_WEIGHTS = 'models/policy_seed_17.pt'
DEFAULT_LOOP_HZ = 20.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--hz', type=float, default=DEFAULT_LOOP_HZ,
        help='policy history rate in Hz (default 20, matching coach). '
             'Training --target-dt is 0.1 s; pass --hz 10 to match that.',
    )
    parser.add_argument(
        '--seconds', type=float, default=0.0,
        help='disarm after this many seconds (0 = until Ctrl+C)',
    )
    add_policy_flags(parser)
    return parser


def add_policy_flags(parser: argparse.ArgumentParser) -> None:
    """Flags shared by ``tools/run_policy.py`` and ``tune_flight.py policy``.

    ``--hz`` / ``--seconds`` are added by the standalone parser and by
    ``tune_flight.common``; do not add them here.
    """
    parser.add_argument(
        '--weights', default=DEFAULT_WEIGHTS,
        help=f'policy checkpoint (default {DEFAULT_WEIGHTS})',
    )
    parser.add_argument(
        '--yolo', default=None, metavar='PATH',
        help='pose weights (default YOLO_POSE_MODEL_PATH / '
             'models/ROBOFLOW_RETRAIN.pt)',
    )
    parser.add_argument(
        '--detector', choices=('yolo_pose', 'gatenet', 'hsv', 'yolo_hybrid'),
        default=None,
        help='detector backend. Default yolo_pose.',
    )
    parser.add_argument(
        '--panel', action='store_true',
        help='live window: camera plus the observation vector the net sees',
    )
    parser.add_argument(
        '--panel-scale', type=float, default=1.0,
        help='scale factor for the --panel window',
    )


def apply_flight_env(args) -> dict:
    """Export flight env *before* ``import config`` / ``setup_components``."""
    applied = {}
    os.environ['FLIGHT_MODE'] = 'policy'
    applied['FLIGHT_MODE'] = 'policy'
    weights = str(getattr(args, 'weights', None) or DEFAULT_WEIGHTS)
    os.environ['POLICY_WEIGHTS'] = weights
    applied['POLICY_WEIGHTS'] = weights
    yolo = getattr(args, 'yolo', None)
    if yolo:
        os.environ['YOLO_POSE_MODEL_PATH'] = str(yolo)
        applied['YOLO_POSE_MODEL_PATH'] = str(yolo)
    hz = getattr(args, 'hz', None)
    if hz is not None:
        os.environ['POLICY_LOOP_HZ'] = repr(float(hz))
        applied['POLICY_LOOP_HZ'] = float(hz)
    seconds = float(getattr(args, 'seconds', 0.0) or 0.0)
    if seconds > 0.0:
        os.environ['RUN_MAX_SECONDS'] = repr(seconds)
        applied['RUN_MAX_SECONDS'] = seconds
    if bool(getattr(args, 'panel', False)):
        os.environ['OBS_PANEL'] = '1'
        applied['OBS_PANEL'] = 1.0
        scale = float(getattr(args, 'panel_scale', 1.0) or 1.0)
        os.environ['OBS_PANEL_SCALE'] = repr(scale)
        applied['OBS_PANEL_SCALE'] = scale
    detector = getattr(args, 'detector', None)
    if detector:
        os.environ['GATE_DETECTOR_BACKEND'] = str(detector)
        applied['GATE_DETECTOR_BACKEND'] = str(detector)
    os.environ['EKF_USE_PNP'] = '0'
    os.environ['VISION_DISPLAY'] = '0'
    os.environ['TAKEOFF_DURATION_S'] = '0'
    os.environ.setdefault('AUTO_RESET_ON_CRASH', '0')
    os.environ.setdefault('CRASH_USE_SIM_ODOMETRY', '0')
    applied['EKF_USE_PNP'] = 0.0
    applied['VISION_DISPLAY'] = 0.0
    applied['TAKEOFF_DURATION_S'] = 0.0
    return applied


def _prewarm_yolo() -> None:
    print('', flush=True)
    print('>>> YOLO pre-warm starting. In the FlightSim window NOW:', flush=True)
    print('>>>   log in and START A RACE (stay on the pad, facing gate 1).',
          flush=True)
    print('', flush=True)
    try:
        from vision_rx import create_gate_detector
        create_gate_detector()
        print('[VISION] pre-warm detector constructed', flush=True)
    except Exception as exc:
        print(f'[VISION] pre-warm skipped: {exc}', flush=True)


def run_from_args(args, *, prewarm: bool = True) -> int:
    applied = apply_flight_env(args)
    print(
        '[POLICY] autonomous  ' +
        '  '.join(f'{k}={v}' for k, v in applied.items()),
        flush=True,
    )
    print(
        '[POLICY] no gamepad, no H/T — Esc / Ctrl+C disarms',
        flush=True,
    )
    if prewarm:
        _prewarm_yolo()
    import main as race_main
    race_main.run_racing()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_from_args(args)


if __name__ == '__main__':
    sys.exit(main())
