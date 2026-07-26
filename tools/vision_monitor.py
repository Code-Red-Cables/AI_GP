#!/usr/bin/env python
"""Drone vision monitor: near-live view of what the training policy sees.

Watches the trainer's episode-dump directory, renders the newest episode as an
animated GIF — camera view with the gate detection overlaid next to the raw HSV
bitmask the detector thresholds on — and serves it on a small auto-refreshing
web page.

Usage:
    venv/bin/python tools/vision_monitor.py \
        --episodes artifacts/runs/<run>/episodes --port 8666

Then open http://localhost:8666 in a browser. The page refreshes itself; a new
episode appears within ~30 s of the collector dumping it (the trainer saves
every 25th episode).
"""
import argparse
import functools
import glob
import http.server
import os
import sys
import threading
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.gate_detector import detect_gate, _build_mask, _make_cfg  # noqa: E402

# Same overrides the StabilizedController uses on the 64x64 obs (baselines.py).
THUMB_CFG = {"min_area": 8.0, "kernel_size": 1, "hole_min_bbox_frac": 0.06}
SCALE = 4  # 64px -> 256px panels


def _render_frame(rgb: np.ndarray, step: int, reward: float, action: np.ndarray) -> np.ndarray:
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    cfg = _make_cfg(THUMB_CFG)
    mask = _build_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), cfg)
    det = detect_gate(bgr, THUMB_CFG)

    h, w = bgr.shape[:2]
    cam = cv2.resize(bgr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)
    msk = cv2.cvtColor(cv2.resize(mask, (w * SCALE, h * SCALE),
                                  interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    if det is not None:
        x, y, bw, bh = det.bbox_px
        cv2.rectangle(cam, (x * SCALE, y * SCALE),
                      ((x + bw) * SCALE, (y + bh) * SCALE), (0, 255, 0), 2)
        cu, cv_ = int(det.center_px[0] * SCALE), int(det.center_px[1] * SCALE)
        cv2.drawMarker(cam, (cu, cv_), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(cam, f"conf {det.confidence:.2f}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(cam, "no det", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)

    panel = np.hstack([cam, msk])
    bar = np.zeros((26, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, f"step {step:3d}  r={reward:+7.2f}  thrust={action[0]:+.2f} "
                     f"roll={action[1]:+.2f} pitch={action[2]:+.2f} yaw={action[3]:+.2f}",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def render_episode_gif(npz_path: str, out_path: str) -> int:
    d = np.load(npz_path)
    imgs, rew, act = d["image"], d["reward"].ravel(), d["action"]
    frames = []
    for i in range(imgs.shape[0]):
        panel = _render_frame(imgs[i], i, float(rew[i]), act[i])
        frames.append(Image.fromarray(panel[..., ::-1]))  # BGR -> RGB
    tmp = out_path + ".tmp"
    frames[0].save(tmp, format="GIF", save_all=True, append_images=frames[1:],
                   duration=100, loop=0)  # 10 fps, matches sim step rate
    os.replace(tmp, out_path)
    return len(frames)


INDEX_HTML = """<!doctype html>
<html><head><title>drone vision monitor</title>
<meta http-equiv="refresh" content="10">
<style>body{{background:#111;color:#ddd;font-family:monospace;text-align:center}}
img{{image-rendering:pixelated;margin-top:8px}}</style></head>
<body>
<h3>latest dumped episode: {name} ({nframes} frames) &mdash; left: camera + detection, right: HSV bitmask</h3>
<div>page auto-refreshes every 10 s; a new episode appears every ~25 collected episodes</div>
<img src="latest.gif?v={stamp}" width="{width}">
</body></html>
"""


def watcher(episodes_dir: str, out_dir: str) -> None:
    last = None
    while True:
        try:
            eps = sorted(glob.glob(os.path.join(episodes_dir, "*.npz")))
            if eps and eps[-1] != last:
                # skip a file mid-write: require it to be at least 2s old
                if time.time() - os.path.getmtime(eps[-1]) > 2.0:
                    n = render_episode_gif(eps[-1], os.path.join(out_dir, "latest.gif"))
                    name = os.path.basename(eps[-1])
                    with open(os.path.join(out_dir, "index.html"), "w") as f:
                        f.write(INDEX_HTML.format(name=name, nframes=n,
                                                  stamp=int(time.time()),
                                                  width=64 * SCALE * 2))
                    print(f"[monitor] rendered {name} ({n} frames)", flush=True)
                    last = eps[-1]
        except Exception as e:
            print(f"[monitor] render error: {e!r}", flush=True)
        time.sleep(5.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, help="trainer episode-dump dir")
    ap.add_argument("--port", type=int, default=8666)
    ap.add_argument("--out", default="_vision_debug/monitor")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "index.html"), "w") as f:
        f.write("<html><body style='background:#111;color:#ddd'>"
                "waiting for first episode dump ...</body></html>")

    threading.Thread(target=watcher, args=(args.episodes, args.out), daemon=True).start()

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=args.out)
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    print(f"[monitor] serving http://localhost:{args.port} (episodes: {args.episodes})",
          flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
