"""Tests for local four-keypoint pose dataset preflight."""

from pathlib import Path

import pytest

from tools.train_gate_pose import validate_pose_dataset


def _write_dataset(root: Path, flip_idx="[1, 0, 3, 2]") -> Path:
    for split in ("train", "valid"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        (root / split / "images" / "gate_001.jpg").touch()
        (root / split / "labels" / "gate_001.txt").write_text(
            "0 0.5 0.5 0.4 0.5 "
            "0.3 0.3 2 0.7 0.3 2 0.3 0.7 2 0.7 0.7 2\n",
            encoding="utf-8",
        )
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        "train: train/images\n"
        "val: valid/images\n"
        "kpt_shape: [4, 3]\n"
        f"flip_idx: {flip_idx}\n"
        "nc: 1\n"
        "names: ['gate']\n",
        encoding="utf-8",
    )
    return yaml_path


def test_pose_preflight_accepts_four_corner_dataset(tmp_path):
    data_yaml = _write_dataset(tmp_path)

    assert validate_pose_dataset(data_yaml) == {
        "train": (1, 1),
        "val": (1, 1),
    }


def test_pose_preflight_rejects_roboflow_identity_flip_mapping(tmp_path):
    data_yaml = _write_dataset(tmp_path, flip_idx="[0, 1, 2, 3]")

    with pytest.raises(ValueError, match="unsafe horizontal"):
        validate_pose_dataset(data_yaml)


def test_pose_preflight_requires_matching_labels(tmp_path):
    data_yaml = _write_dataset(tmp_path)
    (tmp_path / "valid" / "labels" / "gate_001.txt").unlink()

    with pytest.raises(FileNotFoundError, match="matching pairs"):
        validate_pose_dataset(data_yaml)


def test_pose_preflight_rejects_wrong_label_width(tmp_path):
    data_yaml = _write_dataset(tmp_path)
    (tmp_path / "train" / "labels" / "gate_001.txt").write_text(
        "0 0.5 0.5 0.4 0.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected class . box"):
        validate_pose_dataset(data_yaml)
