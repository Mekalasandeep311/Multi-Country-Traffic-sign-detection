"""Contact sheets of video detections: 3 crops per detected class (spread over its detections).
usage: python video_sheet.py <detections.csv> <video> <out_prefix>"""
import collections
import csv
import sys

import cv2
import numpy as np

rows = list(csv.DictReader(open(sys.argv[1])))
by = collections.defaultdict(list)
for r in rows:
    by[r["class_name"]].append(r)
cap = cv2.VideoCapture(sys.argv[2])
tiles = []
for k in sorted(by, key=lambda k: -len(by[k])):
    rs = by[k]
    for r in [rs[int(i)] for i in np.linspace(0, len(rs) - 1, 3)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame"]))
        ok, im = cap.read()
        x0, y0, x1, y1 = map(float, (r["x0"], r["y0"], r["x1"], r["y1"]))
        m = 1.2 * max(x1 - x0, y1 - y0) + 10
        ox, oy = int(max(0, x0 - m)), int(max(0, y0 - m))
        cr = im[oy:int(y1 + m), ox:int(x1 + m)].copy()
        cv2.rectangle(cr, (int(x0) - ox, int(y0) - oy), (int(x1) - ox, int(y1) - oy), (0, 255, 0), 1)
        cr = cv2.resize(cr, (200, 200))
        cv2.rectangle(cr, (0, 0), (200, 30), (0, 0, 0), -1)
        cv2.putText(cr, k[:32], (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
        cv2.putText(cr, f"{float(r['conf']):.2f} t={float(r['time_s']):.0f}s", (2, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1)
        tiles.append(cr)
tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 9)
strips = [np.hstack(tiles[i:i + 9]) for i in range(0, len(tiles), 9)]
for j in range(0, len(strips), 6):
    cv2.imwrite(f"{sys.argv[3]}_{j // 6}.jpg", np.vstack(strips[j:j + 6]))
print(len(by), "classes,", len(tiles), "tiles")
