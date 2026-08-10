"""Tests for detector-triggered raw gate-frame capture."""

import re

import cv2
import numpy as np

import config
from vision_rx import save_gate_capture


def test_gate_capture_preserves_incoming_jpeg_bytes(tmp_path):
    image = np.full((24, 32, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    jpeg = encoded.tobytes()

    path = save_gate_capture(tmp_path, "session", 42, image, jpeg)

    assert path.name == "gate_session_0000000042.jpg"
    assert path.read_bytes() == jpeg


def test_gate_capture_can_encode_an_image_when_jpeg_is_unavailable(tmp_path):
    image = np.zeros((24, 32, 3), dtype=np.uint8)

    path = save_gate_capture(tmp_path, "session", 7, image)

    decoded = cv2.imread(str(path))
    assert decoded is not None
    assert decoded.shape == image.shape


def test_capture_is_rate_limited_to_three_per_second():
    # A flat, uncapped capture dir reached 282k files; three a second is the
    # budget, so the spacing must round-trip to 3 Hz.
    interval = config.GATE_FRAME_CAPTURE_INTERVAL_S
    assert interval > 0.0
    assert round(1.0 / interval) == 3


def test_captures_land_in_a_per_run_subfolder(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'GATE_FRAME_CAPTURE', True)
    monkeypatch.setattr(config, 'GATE_FRAME_CAPTURE_DIR', str(tmp_path))
    from vision_rx import VisionRX

    receiver = VisionRX.__new__(VisionRX)
    VisionRX._init_gate_capture(receiver)

    assert receiver._gate_capture_dir.parent == tmp_path
    assert re.fullmatch(
        r'run_\d{8}_\d{6}', receiver._gate_capture_dir.name
    ), receiver._gate_capture_dir.name
    assert receiver._gate_capture_dir.is_dir()
