"""Tests for local eight-keypoint pose dataset preflight."""

from pathlib import Path

import pytest

from tools.train_gate_pose import (
    POSE_NO_AUGMENTATION,
    POSE_TRAIN_AUGMENTATION,
    build_train_kwargs,
    parse_args,
    validate_pose_dataset,
)

# Outer ring then inner ring, each clockwise from top-left.
_OUTER = "0.3 0.3 2 0.7 0.3 2 0.7 0.7 2 0.3 0.7 2"
_INNER = "0.4 0.4 2 0.6 0.4 2 0.6 0.6 2 0.4 0.6 2"
_LABEL = f"0 0.5 0.5 0.4 0.5 {_OUTER} {_INNER}\n"


def _write_dataset(
    root: Path,
    flip_idx="[1, 0, 3, 2, 5, 4, 7, 6]",
    label=_LABEL,
) -> Path:
    for split in ("train", "valid"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        (root / split / "images" / "gate_001.jpg").touch()
        (root / split / "labels" / "gate_001.txt").write_text(
            label, encoding="utf-8"
        )
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        "train: train/images\n"
        "val: valid/images\n"
        "kpt_shape: [8, 3]\n"
        f"flip_idx: {flip_idx}\n"
        "nc: 1\n"
        "names: ['gate']\n",
        encoding="utf-8",
    )
    return yaml_path


def test_pose_preflight_accepts_eight_keypoint_dataset(tmp_path):
    data_yaml = _write_dataset(tmp_path)

    assert validate_pose_dataset(data_yaml) == {
        "train": (1, 1),
        "val": (1, 1),
    }


def test_pose_preflight_allows_an_unlabelled_corner(tmp_path):
    # Gates leave frame constantly, so a corner written as 0 0 0 is normal
    # and must not be read as a point sitting at the image origin.
    hidden = _LABEL.replace("0.3 0.3 2", "0 0 0", 1)
    data_yaml = _write_dataset(tmp_path, label=hidden)

    assert validate_pose_dataset(data_yaml) == {
        "train": (1, 1),
        "val": (1, 1),
    }


def test_pose_preflight_rejects_a_four_keypoint_dataset(tmp_path):
    data_yaml = _write_dataset(tmp_path)
    data_yaml.write_text(
        data_yaml.read_text(encoding="utf-8").replace(
            "kpt_shape: [8, 3]", "kpt_shape: [4, 3]"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="kpt_shape"):
        validate_pose_dataset(data_yaml)


def test_pose_preflight_rejects_roboflow_identity_flip_mapping(tmp_path):
    # This is what Roboflow actually exports, and it silently teaches the
    # model mirrored corner identities because training flips left-right.
    data_yaml = _write_dataset(tmp_path, flip_idx="[0, 1, 2, 3, 4, 5, 6, 7]")

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


def test_pose_training_rotates_for_race_roll():
    assert POSE_TRAIN_AUGMENTATION["degrees"] == 90.0
    assert POSE_TRAIN_AUGMENTATION["fliplr"] == 0.5
    assert POSE_TRAIN_AUGMENTATION["flipud"] == 0.0


def test_build_train_kwargs_applies_augmentation_and_degrees_override(tmp_path):
    data_yaml = _write_dataset(tmp_path)
    args = parse_args(["--data", str(data_yaml), "--degrees", "0"])
    kwargs = build_train_kwargs(args)

    assert kwargs["degrees"] == 0.0
    assert kwargs["translate"] == POSE_TRAIN_AUGMENTATION["translate"]
    assert kwargs["scale"] == POSE_TRAIN_AUGMENTATION["scale"]
    assert kwargs["fliplr"] == 0.5
    assert kwargs["flipud"] == 0.0


def test_build_train_kwargs_disables_augmentation_when_asked(tmp_path):
    data_yaml = _write_dataset(tmp_path)
    args = parse_args(["--data", str(data_yaml), "--no-augment"])
    kwargs = build_train_kwargs(args)

    assert kwargs["degrees"] == 0.0
    assert kwargs["mosaic"] == 0.0
    assert kwargs["fliplr"] == 0.0
    assert kwargs["translate"] == POSE_NO_AUGMENTATION["translate"]
