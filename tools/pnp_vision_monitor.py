#!/usr/bin/env python
"""Live browser view of the PnP / OpenCV vision pipeline.

``main.py`` (PnP mode) writes annotated camera frames to ``frames/latest.jpg``
(~12 Hz). This tool serves that file on a tiny auto-refreshing page.

Usage (two terminals):

    # 1) fly / perception with overlays
    DEBUG_VISION=1 venv/bin/python main.py

    # 2) browser monitor
    venv/bin/python tools/pnp_vision_monitor.py --port 8666

Then open http://localhost:8666

Legend (PnP mode):
  green box/corners  — YOLO box with a successful PnP solve
  orange box/corners — YOLO box, corners rejected / unsolved (box-only guidance)
  ">>" label         — the gate currently published to the planner / VIO
  grey crosshair     — image centre (ideal aim point)
"""
from __future__ import annotations

import argparse
import functools
import http.server
import os
import time

DEFAULT_DIR = "frames"
DEFAULT_PORT = 8666

INDEX_HTML = """<!doctype html>
<html>
<head>
  <title>AI-GP vision pipeline</title>
  <meta charset="utf-8">
  <style>
    body {{ background:#111; color:#ddd; font-family: ui-monospace, monospace;
           margin:0; padding:16px; text-align:center; }}
    h1 {{ font-size:16px; font-weight:600; margin:0 0 8px; }}
    .meta {{ color:#888; font-size:12px; margin-bottom:12px; }}
    img {{ max-width:100%; height:auto; background:#000; border:1px solid #333; }}
    .legend {{ text-align:left; display:inline-block; font-size:12px; color:#aaa;
               margin-top:12px; line-height:1.5; }}
    .ok {{ color:#6f6; }} .miss {{ color:#f86; }}
  </style>
</head>
<body>
  <h1>AI-GP vision pipeline</h1>
  <div class="meta" id="meta">waiting for frames… (start main.py with DEBUG_VISION=1)</div>
  <div><img id="frame" alt="latest vision frame" width="960"></div>
  <div class="legend">
    <div><span class="ok">green</span> = PnP solved &nbsp;|&nbsp;
         <span style="color:#fc8">orange</span> = YOLO box only &nbsp;|&nbsp;
         <b>&gt;&gt;</b> = published target</div>
    <div>crosshair = image centre &nbsp;|&nbsp; ring stills in <code>frames/frame_*.jpg</code></div>
  </div>
  <script>
    const img = document.getElementById('frame');
    const meta = document.getElementById('meta');
    let lastOk = 0;
    async function tick() {{
      const t = Date.now();
      try {{
        const r = await fetch('latest.jpg?t=' + t, {{ cache: 'no-store' }});
        if (!r.ok) throw new Error('HTTP ' + r.status);
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        const prev = img.src;
        img.src = url;
        if (prev && prev.startsWith('blob:')) URL.revokeObjectURL(prev);
        lastOk = t;
        const age = r.headers.get('X-Frame-Age');
        meta.innerHTML = '<span class="ok">live</span> · ' +
          new Date().toLocaleTimeString() +
          (age != null ? (' · frame age ' + Number(age).toFixed(2) + 's') : '');
      }} catch (e) {{
        const stale = lastOk ? ((t - lastOk) / 1000).toFixed(1) + 's since last frame' : 'no frame yet';
        meta.innerHTML = '<span class="miss">stale</span> · ' + stale +
          ' · run: <code>DEBUG_VISION=1 venv/bin/python main.py</code>';
      }}
    }}
    tick();
    setInterval(tick, 200);  // ~5 Hz UI refresh; source file updates ~12 Hz
  </script>
</body>
</html>
"""


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, debug_dir=DEFAULT_DIR, **kwargs):
        self.debug_dir = os.path.abspath(debug_dir)
        super().__init__(*args, directory=self.debug_dir, **kwargs)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/latest.jpg"):
            path = os.path.join(self.debug_dir, "latest.jpg")
            if not os.path.isfile(path):
                self.send_error(404, "no latest.jpg yet — is DEBUG_VISION=1?")
                return
            age = time.time() - os.path.getmtime(path)
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Age", f"{age:.3f}")
            self.end_headers()
            self.wfile.write(data)
            return
        return super().do_GET()

    def log_message(self, fmt, *args):
        # Quiet — the 5 Hz poll would spam the terminal otherwise.
        if args and isinstance(args[0], str) and "latest.jpg" in args[0]:
            return
        super().log_message(fmt, *args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--dir", default=DEFAULT_DIR, help="overlay directory")
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    handler = functools.partial(Handler, debug_dir=args.dir)
    httpd = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Vision monitor: http://127.0.0.1:{args.port}", flush=True)
    print(f"  watching {os.path.abspath(args.dir)}/latest.jpg", flush=True)
    print("  start flight with: DEBUG_VISION=1 venv/bin/python main.py", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nmonitor stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
