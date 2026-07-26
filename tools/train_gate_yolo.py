"""Train and install the one-class gate model used by the hybrid detector."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_YAML = REPOSITORY_ROOT / "models" / "gate_dataset.yaml"
DEFAULT_DATASET_ROOT = REPOSITORY_ROOT / "datasets" / "gates"
IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_YAML),
        help="Ultralytics detection dataset YAML",
    )
    parser.add_argument(
        "--init-dataset",
        action="store_true",
        help="create the empty one-class dataset folders, then exit",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="pretrained model used only as the training starting point",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--name", default="gate_train", help="Ultralytics run name"
    )
    parser.add_argument(
        "--output",
        default=str(REPOSITORY_ROOT / "models" / "gate_detector.pt"),
        help="where to install trained best.pt",
    )
    return parser.parse_args(argv)


def initialize_dataset(dataset_root=DEFAULT_DATASET_ROOT):
    """Create the expected folders without inventing labels."""
    for split in ("train", "val"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def _files_with_suffix(directory, suffixes):
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def validate_default_dataset(dataset_root=DEFAULT_DATASET_ROOT):
    """Fail before Ultralytics when the local dataset is absent/unlabeled."""
    problems = []
    summary = {}
    for split in ("train", "val"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        images = _files_with_suffix(image_dir, IMAGE_SUFFIXES)
        labels = _files_with_suffix(label_dir, {".txt"})
        matching_labels = {
            path.stem for path in labels
        } & {path.stem for path in images}
        summary[split] = (len(images), len(labels), len(matching_labels))
        if not image_dir.is_dir() or not label_dir.is_dir():
            problems.append(f"{split}: required folders are missing")
        elif not images:
            problems.append(f"{split}: no images found in {image_dir}")
        elif not labels:
            problems.append(f"{split}: no YOLO .txt labels found in {label_dir}")
        elif not matching_labels:
            problems.append(
                f"{split}: image and label filenames have no matching stems"
            )
    if problems:
        details = "\n  - ".join(problems)
        raise FileNotFoundError(
            "A labeled one-class gate dataset is required before training.\n"
            f"  - {details}\n\n"
            f"Expected root: {dataset_root}\n"
            "Expected layout:\n"
            "  datasets/gates/images/train/<frame>.png\n"
            "  datasets/gates/images/val/<frame>.png\n"
            "  datasets/gates/labels/train/<frame>.txt\n"
            "  datasets/gates/labels/val/<frame>.txt\n\n"
            "Each label row must be: 0 center_x center_y width height\n"
            "Label every physical gate separately, especially overlaps.\n"
            "Run with --init-dataset to create the empty folders."
        )
    return summary


def main():
    args = parse_args()
    if args.init_dataset:
        initialize_dataset()
        print(
            f"Created gate dataset folders under: {DEFAULT_DATASET_ROOT}\n"
            "Now add train/validation images and matching YOLO .txt labels.",
            flush=True,
        )
        return 0
    data_path = Path(args.data).resolve()
    if not data_path.is_file():
        print(f"[DATASET ERROR] YAML not found: {data_path}", file=sys.stderr)
        return 2
    if data_path == DEFAULT_DATA_YAML.resolve():
        try:
            summary = validate_default_dataset()
        except FileNotFoundError as exc:
            print(f"[DATASET ERROR]\n{exc}", file=sys.stderr)
            return 2
        print(
            "[DATASET] "
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
            "Install training/runtime dependencies first: "
            "python -m pip install -r requirements.txt"
        ) from exc

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "name": args.name,
    }
    if args.device:
        train_kwargs["device"] = args.device
    result = model.train(**train_kwargs)
    save_dir = Path(getattr(result, "save_dir", "runs/detect/gate_train"))
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(
            f"training completed but best weights were not found: {best_weights}"
        )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, output)
    print(f"Installed custom gate weights: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
