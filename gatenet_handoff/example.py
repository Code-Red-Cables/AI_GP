"""End to end: a camera frame -> the gate's pose in the camera frame.

Run from the directory this file is in:

    pip install onnxruntime opencv-python numpy
    python example.py <frame.jpg>
"""

import sys

import cv2
import numpy as np
import onnxruntime as ort

from aigp.geometry.pnp import is_front_view, solve_gate_pnp
from aigp.perception.cnn_decode import decode_heatmaps, preprocess

NET_SIZE = (320, 180)        # what the network runs at, fixed by the export
CONF = 0.80                  # gate on the weakest corner, not the conf output
GATE_INNER_M = 1.5           # AI-GP gate: the side of the inner square

# Camera intrinsics for the SOURCE image. These are the AI-GP simulator's, for
# a 640x360 frame -- replace them with your own or PnP will be confidently
# wrong. The corners are resolution independent; K is not.
FX = FY = 320.0
CX, CY = 320.0, 180.0


def main(path):
    sess = ort.InferenceSession("gatenet.onnx",
                                providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]

    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"could not read {path}")
    h, w = frame.shape[:2]

    hm, off, conf = sess.run(names, {"image": preprocess(frame, NET_SIZE)})
    corners, score = decode_heatmaps(hm, off, NET_SIZE)

    # normalised -> YOUR pixels. Scaling by 320x180 here is the classic bug:
    # PnP still solves, at roughly double the true range, silently.
    corners_px = (corners[0] * np.array([w, h])).astype(np.float32)

    print(f"{path}  {w}x{h}")
    print("  per-corner scores  " +
          "  ".join(f"{n} {s:.2f}" for n, s in
                    zip(("TL", "TR", "BR", "BL"), score[0])))

    if score[0].min() < CONF:
        n_seen = int((score[0] >= CONF).sum())
        print(f"  no full aperture ({n_seen}/4 corners above {CONF}).")
        if n_seen >= 2:
            seen = corners_px[score[0] >= CONF]
            print(f"  bearing hint only: cx = {seen[:, 0].mean():.1f} px")
        return

    K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
    res = solve_gate_pnp(corners_px, GATE_INNER_M, K)
    if res is None or not is_front_view(res):
        print("  corners found, but no valid front-view PnP solution")
        return

    t = res.T_cg.t
    print(f"  corners px         {np.round(corners_px, 1).tolist()}")
    print(f"  gate in camera     x {t[0]:+.2f}  y {t[1]:+.2f}  z {t[2]:+.2f} m")
    print(f"  range              {np.linalg.norm(t):.2f} m")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "frame.jpg")
