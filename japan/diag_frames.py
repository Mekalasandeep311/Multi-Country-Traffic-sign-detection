"""Diagnose missed signs on screenshots of the detection video: crop the video area, resize to the
native 1280x720, run the model at a very low threshold and draw every box with class + conf.
usage: python diag_frames.py <weights> <in_dir> <out_dir>"""
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

m = YOLO(sys.argv[1])
out = Path(sys.argv[3]); out.mkdir(exist_ok=True)
for p in sorted(Path(sys.argv[2]).glob("*.png")):
    im = cv2.imread(str(p))[150:1020, 102:1890]          # video area below the burnt-in header
    im = cv2.resize(im, (1280, int(1280 * im.shape[0] / im.shape[1])))
    r = m.predict(im, imgsz=1280, conf=0.05, verbose=False)[0]
    print(f"== {p.name}")
    for b, c, k in sorted(zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()), key=lambda t: -t[1]):
        x0, y0, x1, y1 = map(int, b)
        print(f"  {c:.2f} {m.names[int(k)]:<60} box h={y1 - y0}px at ({x0},{y0})")
        col = (0, 255, 0) if c >= 0.5 else (0, 200, 255) if c >= 0.25 else (0, 0, 255)
        cv2.rectangle(im, (x0, y0), (x1, y1), col, 1)
        cv2.putText(im, f"{m.names[int(k)][:22]} {c:.2f}", (x0, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    cv2.imwrite(str(out / (p.stem + "_diag.jpg")), im)
