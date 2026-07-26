"""Train and install the one-class gate model used by the hybrid detector."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(REPOSITORY_ROOT / "models" / "gate_dataset.yaml"),
        help="Ultralytics detection dataset YAML",
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
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"dataset YAML not found: {data_path}")
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


if __name__ == "__main__":
    main()
