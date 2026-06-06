"""Interactive HSV threshold tuner for gate detection (PLAN 8.5).

DEBUG / calibration tool only. Feed it a captured frame:

    & "<repo>/PyAIPilotExample/myenv/Scripts/python.exe" tools/hsv_tuner.py reference/frames/frame_0000.png

Drag the trackbars until only the gate is white in the mask / visible in the
masked view, then press 'q'. The chosen LOWER_HSV / UPPER_HSV are printed — paste
them into vision/gate_detector.py (or its DEFAULT_CFG).
"""
import sys

import cv2
import numpy as np

# Added H2lo/H2hi for the double range (red wrap-around).
# Shared S and V sliders keep the UI from getting too tall/cluttered.
_BARS = [("H1lo", 0, 179), ("H1hi", 10, 179),
         ("H2lo", 170, 179), ("H2hi", 179, 179),
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
        
        # Range 1 (Lower Red)
        lo1 = np.array([g("H1lo"), g("Slo"), g("Vlo")], np.uint8)
        hi1 = np.array([g("H1hi"), g("Shi"), g("Vhi")], np.uint8)
        mask1 = cv2.inRange(hsv, lo1, hi1)

        # Range 2 (Upper Red)
        lo2 = np.array([g("H2lo"), g("Slo"), g("Vlo")], np.uint8)
        hi2 = np.array([g("H2hi"), g("Shi"), g("Vhi")], np.uint8)
        mask2 = cv2.inRange(hsv, lo2, hi2)

        # Combine the masks
        mask = cv2.bitwise_or(mask1, mask2)
        
        masked = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow(win, np.hstack([img, masked]))
        
        if (cv2.waitKey(30) & 0xFF) == ord('q'):
            print(f"LOWER_HSV_1 = {tuple(int(x) for x in lo1)}")
            print(f"UPPER_HSV_1 = {tuple(int(x) for x in hi1)}")
            print(f"LOWER_HSV_2 = {tuple(int(x) for x in lo2)}")
            print(f"UPPER_HSV_2 = {tuple(int(x) for x in hi2)}")
            break
            
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python tools/hsv_tuner.py <image.png>")
        sys.exit(1)
    main(sys.argv[1])