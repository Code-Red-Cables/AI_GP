"""Interactive HSV threshold tuner for gate detection (PLAN 8.5).

DEBUG / calibration tool only. Feed it a captured frame:

    & "<repo>/PyAIPilotExample/myenv/Scripts/python.exe" tools/hsv_tuner.py notes/frames/frame_0000.png

Drag the trackbars until only the gate is white in the mask / visible in the
masked view, then press 'q'. The chosen LOWER_HSV / UPPER_HSV are printed — paste
them into vision/gate_detector.py (or its DEFAULT_CFG).
"""
import sys

import cv2
import numpy as np

_BARS = [("Hlo", 0, 179), ("Hhi", 179, 179),
         ("Slo", 80, 255), ("Shi", 255, 255),
         ("Vlo", 80, 255), ("Vhi", 255, 255)]


def main(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Failed to read {path}")
        return
    if img.shape[:2] != (360, 640):
        img = cv2.resize(img, (640, 360))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    win = "hsv_tuner  (drag bars, 'q' to print + quit)"
    cv2.namedWindow(win)
    for name, val, mx in _BARS:
        cv2.createTrackbar(name, win, val, mx, lambda _x: None)

    while True:
        g = lambda n: cv2.getTrackbarPos(n, win)
        lo = np.array([g("Hlo"), g("Slo"), g("Vlo")], np.uint8)
        hi = np.array([g("Hhi"), g("Shi"), g("Vhi")], np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        masked = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow(win, np.hstack([img, masked]))
        if (cv2.waitKey(30) & 0xFF) == ord('q'):
            print(f"LOWER_HSV = {tuple(int(x) for x in lo)}")
            print(f"UPPER_HSV = {tuple(int(x) for x in hi)}")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python tools/hsv_tuner.py <image.png>")
        sys.exit(1)
    main(sys.argv[1])
