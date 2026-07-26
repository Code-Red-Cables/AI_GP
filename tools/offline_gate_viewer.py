"""Run gate detection/tracking/navigation without launching the simulator.

Examples:
    python tools/offline_gate_viewer.py frames
    python tools/offline_gate_viewer.py flight.mp4 --video-out annotated.mp4
    python tools/offline_gate_viewer.py frame.jpg --show --save-dir debug_frames
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.gate_detector import OrangeGateDetector, draw_detection  # noqa: E402
from vision.gate_tracker import GateTracker  # noqa: E402
from vision.navigation import GateNavigator  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def image_paths(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if path.is_dir():
        return sorted(
            item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES
        )
    return []


def compose_debug(frame, mask, overlay):
    colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if colored_mask.shape[:2] != overlay.shape[:2]:
        colored_mask = cv2.resize(colored_mask, (overlay.shape[1], overlay.shape[0]))
    return np.hstack([frame, colored_mask, overlay])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="image, image folder, or video")
    parser.add_argument("--show", action="store_true", help="open a live OpenCV window")
    parser.add_argument("--save-dir", type=Path, help="save annotated PNGs and masks")
    parser.add_argument("--video-out", type=Path, help="record the three-panel debug view")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    detector = OrangeGateDetector()
    tracker = GateTracker()
    navigator = GateNavigator()
    paths = image_paths(args.input)
    capture = None if paths else cv2.VideoCapture(str(args.input))
    if not paths and (capture is None or not capture.isOpened()):
        parser.error(f"cannot open {args.input}")
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    index = 0
    detected = 0
    total_ms = []
    while True:
        if paths:
            if index >= len(paths):
                break
            frame = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
            name = paths[index].stem
        else:
            ok, frame = capture.read()
            if not ok:
                break
            name = f"frame_{index:06d}"
        if frame is None:
            index += 1
            continue

        result = detector.detect(frame)
        tracked = tracker.update(result)
        command = navigator.update(tracked, time.monotonic())
        debug = detector.last_debug
        overlay = draw_detection(
            frame, tracked, debug, state=command.state.value, command=command
        )
        mask = debug.mask if debug else np.zeros(frame.shape[:2], dtype=np.uint8)
        panel = compose_debug(frame, mask, overlay)

        if tracked is not None and tracked.found and not tracked.predicted:
            detected += 1
        if debug:
            total_ms.append(debug.timings_ms.get("total", 0.0))
        if args.save_dir:
            cv2.imwrite(str(args.save_dir / f"{name}_overlay.png"), overlay)
            cv2.imwrite(str(args.save_dir / f"{name}_mask.png"), mask)
        if args.video_out:
            if writer is None:
                args.video_out.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(args.video_out),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30.0,
                    (panel.shape[1], panel.shape[0]),
                )
            writer.write(panel)
        if args.show:
            cv2.imshow("raw | HSV mask | gate debug", panel)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        index += 1
        if args.max_frames and index >= args.max_frames:
            break

    if capture is not None:
        capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    mean_ms = sum(total_ms) / len(total_ms) if total_ms else 0.0
    print(
        f"processed={index} measured_detections={detected} "
        f"mean_detector_ms={mean_ms:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
