"""Mine hard negatives from sign-free Singapore frames (singapore_Hardmining/).

Every detection on these frames is a false positive. Outputs (hardneg/):
  detections.csv           frame, class, conf, box  (conf >= MINE_CONF)
  summary.txt              FP counts per class and FP-frames at several thresholds
  sheet_top.jpg            crops of the 80 most confident FPs (to check no real sign is in the set)
  split.json               train / holdout frames (holdout = last contiguous HOLDOUT share)

usage: python mine_hardneg.py [weights]
"""
import collections
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent
SRC = ROOT / "singapore_Hardmining"
OUT = ROOT / "hardneg"
MINE_CONF = 0.15
HOLDOUT = 0.2


def run(weights, frames, out_csv=None):
    model = YOLO(str(weights))
    rows = []
    for f in frames:
        im = cv2.imread(str(f))
        imgsz = int(np.ceil(max(im.shape[:2]) / 32) * 32)
        r = model.predict(im, imgsz=imgsz, conf=MINE_CONF, verbose=False)[0]
        for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            rows.append([f.name, model.names[int(k)], round(c, 3), *[round(v, 1) for v in b]])
    if out_csv:
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "class", "conf", "x0", "y0", "x1", "y1"])
            w.writerows(rows)
    return rows


def summarize(rows, n_frames, title):
    lines = [f"== {title}: {n_frames} frames"]
    for t in (0.15, 0.25, 0.35, 0.5):
        sel = [r for r in rows if r[2] >= t]
        lines.append(f"conf>={t}: {len(sel)} FP boxes, {len({r[0] for r in sel})} frames with FP "
                     f"({len(sel) / n_frames:.2f} FP/frame)")
    lines.append("top FP classes (conf>=0.25): " + ", ".join(
        f"{c} {n}" for c, n in collections.Counter(r[1] for r in rows if r[2] >= 0.25).most_common(12)))
    return "\n".join(lines)


def main(weights):
    OUT.mkdir(exist_ok=True)
    frames = sorted(SRC.glob("*.jpg"))
    n_hold = int(round(HOLDOUT * len(frames)))
    split = {"train": [f.name for f in frames[:-n_hold]], "holdout": [f.name for f in frames[-n_hold:]]}
    (OUT / "split.json").write_text(json.dumps(split, indent=1))

    rows = run(weights, frames, OUT / "detections.csv")
    hold = set(split["holdout"])
    text = "\n".join([summarize(rows, len(frames), "all"),
                      summarize([r for r in rows if r[0] in hold], len(hold), "holdout (baseline)")])
    (OUT / "summary.txt").write_text(text + "\n")
    print(text)

    top = sorted(rows, key=lambda r: -r[2])[:80]
    tiles = []
    for f, cls, c, x0, y0, x1, y1 in top:
        im = cv2.imread(str(SRC / f))
        m = 1.0 * max(x1 - x0, y1 - y0) + 8
        crop = im[int(max(0, y0 - m)):int(y1 + m), int(max(0, x0 - m)):int(x1 + m)].copy()
        ox, oy = int(max(0, x0 - m)), int(max(0, y0 - m))
        cv2.rectangle(crop, (int(x0) - ox, int(y0) - oy), (int(x1) - ox, int(y1) - oy), (0, 0, 255), 1)
        crop = cv2.resize(crop, (160, 160))
        cv2.putText(crop, f"{c:.2f} {cls[:20]}", (2, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        cv2.putText(crop, f[-10:-4], (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        tiles.append(crop)
    if tiles:
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 10)
        cv2.imwrite(str(OUT / "sheet_top.jpg"), np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "sg_signs_v2" / "weights" / "best.pt")
