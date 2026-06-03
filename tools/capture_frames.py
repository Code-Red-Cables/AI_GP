"""Capture FPV frames from the sim's UDP vision stream for HSV calibration (PLAN 8.5).

DEBUG / calibration tool only — never part of a timed run. Launch the sim, get into
a race so video is streaming, then run:

    & "<repo>/PyAIPilotExample/myenv/Scripts/python.exe" tools/capture_frames.py [count] [out_dir]

Saves reassembled JPEG frames as PNGs you can feed to tools/hsv_tuner.py and the
gate_detector offline self-test.
"""
import os
import socket
import struct
import sys

import cv2
import numpy as np

UDP_IP, UDP_PORT = "0.0.0.0", 5600
HEADER_FMT = "<IHHIIQ"   # frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns


def main(count=50, out_dir="reference/frames"):
    os.makedirs(out_dir, exist_ok=True)
    header_sz = struct.calcsize(HEADER_FMT)
    frames = {}

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening on {UDP_IP}:{UDP_PORT}; saving up to {count} frames to {out_dir}/", flush=True)

    saved = 0
    while saved < count:
        packet, _ = sock.recvfrom(65536)
        header, payload = packet[:header_sz], packet[header_sz:]
        frame_id, chunk_id, total_chunks, _jpeg, _psz, _t = struct.unpack(HEADER_FMT, header)

        f = frames.setdefault(frame_id, {"chunks": {}, "total": total_chunks})
        f["chunks"][chunk_id] = payload

        if len(f["chunks"]) == f["total"]:
            buf = bytearray()
            ok = True
            for i in range(f["total"]):
                if i not in f["chunks"]:
                    ok = False
                    break
                buf.extend(f["chunks"][i])
            del frames[frame_id]
            if not ok:
                continue
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            path = os.path.join(out_dir, f"frame_{saved:04d}.png")
            cv2.imwrite(path, img)
            saved += 1
            print(f"  saved {path}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    cnt = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    od = sys.argv[2] if len(sys.argv) > 2 else "reference/frames"
    main(cnt, od)
