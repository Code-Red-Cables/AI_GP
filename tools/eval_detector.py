import os
import glob
import cv2
import sys

# ensure we can import vision
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision.gate_detector import detect_gates, draw_detection

def main():
    image_paths = []
    # Running from repo root
    for d in ['frames', '_vision_debug']:
        if os.path.exists(d):
            image_paths.extend(glob.glob(f"{d}/*.png"))
            image_paths.extend(glob.glob(f"{d}/*.jpg"))

    if not image_paths:
        print("No frames found to evaluate.")
        return

    total = len(image_paths)
    detected = 0
    four_corners = 0
    sum_conf = 0.0

    html_lines = [
        "<html><head><title>Detector Eval</title></head><body>",
        "<h1>Detector Evaluation</h1>",
        "<div style='display:flex; flex-wrap:wrap;'>"
    ]

    print(f"Evaluating {total} frames...")

    os.makedirs("eval_out", exist_ok=True)

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        
        dets = detect_gates(img)
        best = dets[0] if dets else None
        
        out_img = draw_detection(img, best)
        # Use simple basename for html
        base = os.path.basename(path)
        out_name = f"eval_out/{i:04d}_{base}"
        cv2.imwrite(out_name, out_img)

        conf_str = f"conf: {best.confidence:.2f}" if best else "none"
        html_lines.append(f"<div style='margin:5px;'><img src='{out_name}' width='320'><br>{base}<br>{conf_str}</div>")
        
        if best:
            detected += 1
            if best.corners_px is not None and len(best.corners_px) == 4:
                four_corners += 1
            sum_conf += best.confidence

    html_lines.append("</div></body></html>")
    with open("eval_results.html", "w") as f:
        f.write("\n".join(html_lines))

    det_rate = (detected / total) * 100 if total > 0 else 0
    corner_rate = (four_corners / detected) * 100 if detected > 0 else 0
    mean_conf = sum_conf / detected if detected > 0 else 0

    stats = (f"frames: {total}, detection rate: {det_rate:.1f}%, "
             f"4-corner rate: {corner_rate:.1f}%, mean confidence: {mean_conf:.3f}")
    
    print(stats)
    
if __name__ == "__main__":
    main()
