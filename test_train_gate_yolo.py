"""Tests for the gate-dataset training preflight."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import train_gate_yolo
from tools.train_gate_yolo import initialize_dataset, validate_default_dataset


def test_initialize_dataset_creates_all_split_folders(tmp_path):
    initialize_dataset(tmp_path)

    for split in ("train", "val"):
        assert (tmp_path / "images" / split).is_dir()
        assert (tmp_path / "labels" / split).is_dir()


def test_preflight_explains_that_labels_are_required(tmp_path):
    initialize_dataset(tmp_path)

    with pytest.raises(FileNotFoundError, match="no images found"):
        validate_default_dataset(tmp_path)


def test_preflight_accepts_matching_images_and_labels(tmp_path):
    initialize_dataset(tmp_path)
    for split in ("train", "val"):
        (tmp_path / "images" / split / "gate_001.png").touch()
        (tmp_path / "labels" / split / "gate_001.txt").write_text(
            "0 0.5 0.5 0.4 0.6\n",
            encoding="utf-8",
        )

    summary = validate_default_dataset(tmp_path)

    assert summary == {"train": (1, 1, 1), "val": (1, 1, 1)}


def test_dataset_yaml_paths_are_relative_to_models_directory():
    yaml_text = (
        Path(__file__).resolve().parent / "models" / "gate_dataset.yaml"
    ).read_text(encoding="utf-8")

    assert "path:" not in yaml_text
    assert "train: ../datasets/gates/images/train" in yaml_text
    assert "val: ../datasets/gates/images/val" in yaml_text


def test_main_stops_cleanly_before_ultralytics_without_dataset(
    monkeypatch, tmp_path, capsys
):
    empty_yaml = tmp_path / "gate_dataset.yaml"
    empty_yaml.write_text(
        "train: images/train\nval: images/val\nnames:\n  0: gate\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(train_gate_yolo, "DEFAULT_DATA_YAML", empty_yaml)
    def missing_dataset():
        raise FileNotFoundError("labels are missing")

    monkeypatch.setattr(
        train_gate_yolo, "validate_default_dataset", missing_dataset
    )
    monkeypatch.setattr(
        train_gate_yolo,
        "parse_args",
        lambda: SimpleNamespace(
            init_dataset=False,
            data=str(empty_yaml),
        ),
    )

    assert train_gate_yolo.main() == 2
    assert "[DATASET ERROR]" in capsys.readouterr().err
