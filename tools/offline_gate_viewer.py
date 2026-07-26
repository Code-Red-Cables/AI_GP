"""Inspect and tune deterministic gate navigation without the simulator.

Examples:
    python tools/offline_gate_viewer.py frames --show --step
    python tools/offline_gate_viewer.py flight.mp4 --video-out annotated.mp4
    python tools/offline_gate_viewer.py frame.jpg --tune-hsv --show-hsv

Interactive keys:
    n / space / right arrow: next frame
    p / left arrow: previous frame (tracker is reset)
    r: reprocess the current frame after changing HSV trackbars
    q / escape: quit
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.gate_detector import (  # noqa: E402
    GateVisionConfig,
    OrangeGateDetector,
    draw_detection,
)
from vision.gate_tracker import GateTracker  # noqa: E402
from vision.navigation import GateNavigator  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
WINDOW_NAME = "original | raw mask | cleaned mask | gate debug"
TUNING_WINDOW = "orange HSV tuning"


def image_paths(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if path.is_dir():
        return sorted(
            item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES
        )
    return []


def _label(image: np.ndarray, text: str) -> np.ndarray:
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (min(labeled.shape[1], 230), 27), (0, 0, 0), -1)
    cv2.putText(
        labeled,
        text,
        (7, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return labeled


def _mask_panel(mask: np.ndarray, shape: tuple[int, int], label: str) -> np.ndarray:
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return _label(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), label)


def compose_debug(
    frame: np.ndarray,
    raw_mask: np.ndarray,
    cleaned_mask: np.ndarray,
    overlay: np.ndarray,
    hsv: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a four-panel view, optionally with H/S/V channels underneath."""
    shape = overlay.shape[:2]
    original = frame
    if original.shape[:2] != shape:
        original = cv2.resize(original, (shape[1], shape[0]))
    top = np.hstack(
        [
            _label(original, "original BGR"),
            _mask_panel(raw_mask, shape, "raw orange mask"),
            _mask_panel(cleaned_mask, shape, "cleaned mask"),
            _label(overlay, "candidates / tracker / control"),
        ]
    )
    if hsv is None:
        return top
    if hsv.shape[:2] != shape:
        hsv = cv2.resize(hsv, (shape[1], shape[0]))
    channels = [
        _mask_panel(channel, shape, name)
        for channel, name in zip(cv2.split(hsv), ("H channel", "S channel", "V channel"))
    ]
    channels.append(np.zeros_like(channels[0]))
    return np.vstack([top, np.hstack(channels)])


def fit_video_frame(panel: np.ndarray, maximum_width: int = 1920) -> np.ndarray:
    """Keep debug videos within broadly supported encoder dimensions."""
    if panel.shape[1] <= maximum_width:
        return panel
    scale = maximum_width / float(panel.shape[1])
    width = max(2, int(round(panel.shape[1] * scale)) // 2 * 2)
    height = max(2, int(round(panel.shape[0] * scale)) // 2 * 2)
    return cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)


def open_video_writer(
    path: Path, fps: float, frame_size: tuple[int, int]
) -> cv2.VideoWriter:
    """Open a portable writer, accounting for platform codec differences."""
    if path.suffix.lower() == ".avi":
        codecs = ("MJPG", "XVID")
    elif sys.platform == "darwin":
        codecs = ("avc1", "mp4v")
    else:
        codecs = ("mp4v", "avc1")
    for codec in codecs:
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), fps, frame_size
        )
        if writer.isOpened():
            print(f"video_codec={codec} output={path}")
            return writer
        writer.release()
    raise RuntimeError(
        f"cannot create video {path}; try an .avi output for MJPG fallback"
    )


def _noop(_value: int) -> None:
    pass


def create_hsv_trackbars(config: GateVisionConfig) -> None:
    cv2.namedWindow(TUNING_WINDOW, cv2.WINDOW_NORMAL)
    ranges = list(config.hsv_ranges)
    while len(ranges) < 2:
        ranges.append(ranges[0])
    for range_index, (lower, upper) in enumerate(ranges[:2], start=1):
        for channel, maximum, low, high in zip(
            "HSV", (179, 255, 255), lower, upper
        ):
            cv2.createTrackbar(
                f"{range_index}{channel} low",
                TUNING_WINDOW,
                int(low),
                maximum,
                _noop,
            )
            cv2.createTrackbar(
                f"{range_index}{channel} high",
                TUNING_WINDOW,
                int(high),
                maximum,
                _noop,
            )


def config_from_trackbars(base: GateVisionConfig) -> GateVisionConfig:
    ranges = []
    for range_index in (1, 2):
        lower = tuple(
            cv2.getTrackbarPos(f"{range_index}{channel} low", TUNING_WINDOW)
            for channel in "HSV"
        )
        upper = tuple(
            cv2.getTrackbarPos(f"{range_index}{channel} high", TUNING_WINDOW)
            for channel in "HSV"
        )
        ranges.append((lower, upper))
    return replace(base, hsv_ranges=tuple(ranges))


def _read_frame(
    index: int,
    paths: list[Path],
    capture: Optional[cv2.VideoCapture],
) -> tuple[Optional[np.ndarray], str]:
    if paths:
        if not 0 <= index < len(paths):
            return None, ""
        return cv2.imread(str(paths[index]), cv2.IMREAD_COLOR), paths[index].stem
    assert capture is not None
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    return (frame if ok else None), f"frame_{index:06d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="image, image folder, or video")
    parser.add_argument("--show", action="store_true", help="open the debug window")
    parser.add_argument(
        "--step", action="store_true", help="wait for a key between frames"
    )
    parser.add_argument(
        "--tune-hsv",
        action="store_true",
        help="add live HSV range trackbars (press r to reprocess)",
    )
    parser.add_argument(
        "--show-hsv", action="store_true", help="show H, S, and V channel panels"
    )
    parser.add_argument("--save-dir", type=Path, help="save overlays and both masks")
    parser.add_argument("--video-out", type=Path, help="record the debug panels")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--display-scale",
        type=float,
        default=0.65,
        help="window scale only; saved images/video retain full resolution",
    )
    args = parser.parse_args()
    if args.step or args.tune_hsv:
        args.show = True

    base_config = GateVisionConfig()
    detector = OrangeGateDetector(base_config)
    tracker = GateTracker()
    navigator = GateNavigator()
    paths = image_paths(args.input)
    capture = None if paths else cv2.VideoCapture(str(args.input))
    if not paths and (capture is None or not capture.isOpened()):
        parser.error(f"cannot open {args.input}")
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    source_fps = (
        30.0
        if capture is None
        else float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    )
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        source_fps = 30.0
    frame_count = (
        len(paths)
        if paths
        else int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    )
    if frame_count <= 0:
        frame_count = 2**31 - 1

    if args.show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if args.tune_hsv:
        create_hsv_trackbars(base_config)
        print("HSV tuning enabled: adjust sliders, then press r to reprocess.")

    writer = None
    index = 0
    unique_processed = 0
    measured_detections = 0
    detector_times: list[float] = []
    tracker_times: list[float] = []
    total_times: list[float] = []
    last_written_index = -1
    while index < frame_count:
        frame, name = _read_frame(index, paths, capture)
        if frame is None:
            break
        if args.tune_hsv:
            tuned_config = config_from_trackbars(base_config)
            if tuned_config != detector.config:
                detector = OrangeGateDetector(tuned_config)
                tracker.reset()
                navigator.reset()

        timestamp = index / source_fps
        processing_started = time.perf_counter()
        hint = tracker.hint(timestamp)
        raw_detection = detector.detect(frame, hint=hint, timestamp=timestamp)
        tracker_started = time.perf_counter()
        tracked = tracker.update(raw_detection, timestamp=timestamp)
        tracker_ms = (time.perf_counter() - tracker_started) * 1000.0
        command = navigator.update(tracked, timestamp)
        total_ms = (time.perf_counter() - processing_started) * 1000.0

        debug = detector.last_debug
        overlay = draw_detection(
            frame,
            tracked,
            debug,
            state=command.state.value,
            command=command,
            raw_detection=raw_detection,
            total_time_ms=total_ms,
        )
        empty = np.zeros(frame.shape[:2], dtype=np.uint8)
        raw_mask = debug.raw_mask if debug else empty
        cleaned_mask = debug.cleaned_mask if debug else empty
        panel = compose_debug(
            frame,
            raw_mask,
            cleaned_mask,
            overlay,
            debug.hsv if debug and args.show_hsv else None,
        )

        measured = bool(
            tracked is not None and tracked.found and not tracked.predicted
        )
        if measured:
            measured_detections += 1
        detector_ms = (
            debug.timings_ms.get("total", 0.0) if debug is not None else 0.0
        )
        detector_times.append(detector_ms)
        tracker_times.append(tracker_ms)
        total_times.append(total_ms)
        unique_processed += 1
        tracked_method = tracked.method if tracked is not None else "none"
        tracked_confidence = tracked.confidence if tracked is not None else 0.0
        print(
            f"frame={index:06d} raw={raw_detection.method:<20} "
            f"tracked={tracked_method:<20} confidence={tracked_confidence:.3f} "
            f"state={command.state.value:<18} detector={detector_ms:6.2f}ms "
            f"tracker={tracker_ms:5.2f}ms total={total_ms:6.2f}ms"
        )

        if args.save_dir:
            cv2.imwrite(str(args.save_dir / f"{name}_overlay.png"), overlay)
            cv2.imwrite(str(args.save_dir / f"{name}_raw_mask.png"), raw_mask)
            cv2.imwrite(
                str(args.save_dir / f"{name}_cleaned_mask.png"), cleaned_mask
            )
        if args.video_out and index > last_written_index:
            video_panel = fit_video_frame(panel)
            if writer is None:
                args.video_out.parent.mkdir(parents=True, exist_ok=True)
                writer = open_video_writer(
                    args.video_out,
                    source_fps,
                    (video_panel.shape[1], video_panel.shape[0]),
                )
            writer.write(video_panel)
            last_written_index = index

        key = -1
        if args.show:
            display_scale = max(0.1, args.display_scale)
            shown = cv2.resize(
                panel,
                None,
                fx=display_scale,
                fy=display_scale,
                interpolation=cv2.INTER_AREA,
            )
            cv2.imshow(WINDOW_NAME, shown)
            key = cv2.waitKey(0 if (args.step or args.tune_hsv) else 1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("p"), 81) and index > 0:
                index -= 1
                tracker.reset()
                navigator.reset()
                continue
            if key == ord("r"):
                tracker.reset()
                navigator.reset()
                continue

        index += 1
        if args.max_frames and unique_processed >= args.max_frames:
            break

    final_hsv_ranges = (
        config_from_trackbars(detector.config).hsv_ranges
        if args.tune_hsv
        else None
    )
    if capture is not None:
        capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    print(
        f"processed={unique_processed} measured_detections={measured_detections} "
        f"mean_detector_ms={mean(detector_times):.2f} "
        f"mean_tracker_ms={mean(tracker_times):.3f} "
        f"mean_total_ms={mean(total_times):.2f}"
    )
    if final_hsv_ranges is not None:
        print(f"final_hsv_ranges={final_hsv_ranges}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
