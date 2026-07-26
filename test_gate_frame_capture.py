"""Tests for detector-triggered raw gate-frame capture."""

import cv2
import numpy as np

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
