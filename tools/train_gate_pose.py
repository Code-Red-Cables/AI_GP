"""Train the eight-keypoint gate pose model locally without a hosted API."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    REPOSITORY_ROOT / "datasets" / "AIGP_8keypoints.v1i.yolov8" / "data.yaml"
)
# Outer ring then inner ring, each clockwise from top-left.
KEYPOINT_COUNT = 8
EXPECTED_KEYPOINT_SHAPE = [KEYPOINT_COUNT, 3]
# Mirroring swaps TL<->TR and BR<->BL inside each clockwise ring.
EXPECTED_FLIP_INDEX = [1, 0, 3, 2, 5, 4, 7, 6]
LABEL_FIELD_COUNT = 5 + KEYPOINT_COUNT * 3
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train one-class YOLO26 pose weights for the eight gate corners: "
            "outer then inner ring, each clockwise from top-left."
        )
    )
    parser.add_argument("--data", default=str(DEFAULT_DATASET))
    parser.add_argument("--model", default="yolo26s-pose.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", default="gate_pose_train")
    parser.add_argument(
        "--output",
        default=str(REPOSITORY_ROOT / "models" / "gate_pose.pt"),
    )
    return parser.parse_args(argv)


def _resolve_split(data_yaml: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (data_yaml.parent / path).resolve()


def _validate_pose_label(label_path: Path) -> None:
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != LABEL_FIELD_COUNT:
            raise ValueError(
                f"{label_path}:{line_number} has {len(fields)} values; "
                f"expected class + box + {KEYPOINT_COUNT} "
                "(x, y, visibility) keypoints"
            )
        values = [float(field) for field in fields]
        if int(values[0]) != 0 or values[0] != 0:
            raise ValueError(
                f"{label_path}:{line_number} must use gate class 0"
            )
        if any(value < 0.0 or value > 1.0 for value in values[1:5]):
            raise ValueError(
                f"{label_path}:{line_number} has a box outside [0, 1]"
            )
        for keypoint in range(KEYPOINT_COUNT):
            offset = 5 + keypoint * 3
            x, y, visibility = values[offset: offset + 3]
            # An unlabelled corner is written as 0 0 0, so only points that
            # claim to exist need to sit inside the frame. Roughly a fifth of
            # this set is unlabelled, since gates leave the frame constantly.
            if visibility != 0.0 and not (
                0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
            ):
                raise ValueError(
                    f"{label_path}:{line_number} keypoint {keypoint} "
                    "is outside [0, 1]"
                )
            if visibility not in (0.0, 1.0, 2.0):
                raise ValueError(
                    f"{label_path}:{line_number} keypoint {keypoint} "
                    f"has invalid visibility {visibility}"
                )


def validate_pose_dataset(data_yaml: Path) -> dict[str, tuple[int, int]]:
    """Validate schema, keypoint semantics, and image/label pairing."""
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"pose dataset YAML was not found: {data_yaml}\n"
            "Expected the Roboflow export at "
            f"{DEFAULT_DATASET.parent}"
        )
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    if list(payload.get("kpt_shape", [])) != EXPECTED_KEYPOINT_SHAPE:
        raise ValueError(
            "data.yaml must declare kpt_shape: [8, 3] for eight gate corners"
        )
    if list(payload.get("flip_idx", [])) != EXPECTED_FLIP_INDEX:
        raise ValueError(
            "data.yaml has an unsafe horizontal keypoint mapping. Set "
            "flip_idx: [1, 0, 3, 2, 5, 4, 7, 6] so each ring swaps "
            "TL/TR and BR/BL under a left-right flip."
        )
    names = payload.get("names", [])
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]
    if [str(name).strip().lower() for name in names] != ["gate"]:
        raise ValueError("pose dataset must contain exactly one class: gate")

    summary = {}
    for split in ("train", "val"):
        value = payload.get(split)
        if not value:
            raise ValueError(f"data.yaml is missing the {split!r} split")
        image_dir = _resolve_split(data_yaml, str(value))
        label_dir = image_dir.parent / "labels"
        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ) if image_dir.is_dir() else []
        labels = sorted(label_dir.glob("*.txt")) if label_dir.is_dir() else []
        matched = {path.stem for path in images} & {
            path.stem for path in labels
        }
        if not images or not labels or len(matched) != len(images):
            raise FileNotFoundError(
                f"{split}: expected paired images and labels under "
                f"{image_dir.parent.parent}; found {len(images)} images, "
                f"{len(labels)} labels, and {len(matched)} matching pairs"
            )
        for label_path in labels:
            _validate_pose_label(label_path)
        summary[split] = (len(images), len(labels))
    return summary


def main(argv=None):
    args = parse_args(argv)
    data_yaml = Path(args.data).resolve()
    try:
        summary = validate_pose_dataset(data_yaml)
    except (FileNotFoundError, ValueError, OSError, yaml.YAMLError) as exc:
        print(f"[DATASET ERROR] {exc}", file=sys.stderr)
        return 2
    print(
        "[POSE DATASET] "
        + " ".join(
            f"{split}=images:{counts[0]} labels:{counts[1]}"
            for split, counts in summary.items()
        ),
        flush=True,
    )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Install dependencies first: "
            "python -m pip install -r requirements.txt"
        ) from exc

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "name": args.name,
        "fliplr": 0.5,
    }
    if args.device:
        train_kwargs["device"] = args.device
    result = model.train(**train_kwargs)
    save_dir = Path(
        getattr(result, "save_dir", "runs/pose/gate_pose_train")
    )
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(
            "training finished but best weights were not found at "
            f"{best_weights}"
        )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, output)
    print(
        f"Installed {KEYPOINT_COUNT}-keypoint pose weights: {output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
