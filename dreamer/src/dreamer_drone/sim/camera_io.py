"""Camera transport: a background UDP thread that reassembles chunked JPEG frames
(header `<IHHIIQ>`, 24 bytes) into the latest decoded BGR frame.

Adapted from `../../vision_rx.py`. It keeps only the newest complete frame plus dup/drop
statistics; stale partial frames are pruned. `get_latest()` returns a causal snapshot —
never a future frame. Decoding of the gate detector is NOT done here (that is the env's
vision step) to keep the transport pure.
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cv2
    _HAVE_CV2 = True
except Exception:  # pragma: no cover
    _HAVE_CV2 = False

_HEADER_FMT = "<IHHIIQ"
_HEADER_SZ = struct.calcsize(_HEADER_FMT)  # 24


@dataclass
class Frame:
    image_bgr: np.ndarray
    frame_id: int
    sim_time_ns: int
    recv_wall_s: float


class CameraIO:
    def __init__(self, bind: str, port: int, max_partial: int = 30):
        self.bind = bind
        self.port = port
        self.max_partial = max_partial
        self._lock = threading.Lock()
        self._latest: Optional[Frame] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        # stats
        self.frames_completed = 0
        self.frames_dropped = 0        # incomplete frames pruned
        self.duplicates = 0            # re-sent packets for an already-completed frame
        self._last_frame_id = -1
        self._last_completed_id = -1   # high-water mark to skip re-decoding duplicates
        # the sim re-transmits each frame many times (~38x, measured); a backward jump
        # larger than this is treated as an episode/stream reset, not a duplicate.
        self._reset_gap = 1000

    def start(self) -> "CameraIO":
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def get_latest(self) -> Optional[Frame]:
        with self._lock:
            return self._latest

    def age(self) -> float:
        with self._lock:
            f = self._latest
        return (time.time() - f.recv_wall_s) if f else float("inf")

    def _loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind((self.bind, self.port))
        sock.settimeout(1.0)
        partial: dict[int, dict] = {}
        try:
            while self.running:
                try:
                    packet, _ = sock.recvfrom(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if len(packet) < _HEADER_SZ:
                    continue
                (frame_id, chunk_id, total_chunks, jpeg_size,
                 payload_size, sim_time_ns) = struct.unpack(_HEADER_FMT, packet[:_HEADER_SZ])
                payload = packet[_HEADER_SZ:]

                # Skip packets for a frame we already completed (the sim re-sends each frame
                # ~38x). A large backward jump means the stream reset, so accept it.
                if (frame_id <= self._last_completed_id
                        and (self._last_completed_id - frame_id) < self._reset_gap):
                    self.duplicates += 1
                    continue
                if frame_id < self._last_completed_id:  # reset detected
                    self._last_completed_id = -1

                slot = partial.setdefault(frame_id, {"chunks": {}, "total": total_chunks})
                slot["chunks"][chunk_id] = payload

                if len(slot["chunks"]) == slot["total"]:
                    ok = all(i in slot["chunks"] for i in range(slot["total"]))
                    if ok:
                        jpeg = b"".join(slot["chunks"][i] for i in range(slot["total"]))
                        self._decode_and_store(frame_id, sim_time_ns, jpeg)
                    del partial[frame_id]

                if len(partial) > self.max_partial:
                    self.frames_dropped += 1
                    del partial[min(partial)]
        finally:
            sock.close()

    def _decode_and_store(self, frame_id: int, sim_time_ns: int, jpeg: bytes) -> None:
        if not _HAVE_CV2:
            return
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return
        self._last_completed_id = frame_id   # RX-thread only; gates duplicate skipping
        with self._lock:
            self._last_frame_id = frame_id
            self.frames_completed += 1
            self._latest = Frame(img, frame_id, sim_time_ns, time.time())
