"""Per-class confidence thresholds for detect.py, measured on the real full-frame val set.

The val frames (jp_eval_full/all: 1920x1080 real frames with all known labels) are shrunk to the
video size (1280 wide) so signs have the size they have in the dashcam video. Predictions (conf >=
0.05) are matched to labels (same class, IoU >= 0.5). For every class with >= MIN_GT labels the
threshold with the best F-beta (beta 1.5: recall counts more than precision, missing a sign is worse
than a rare extra box) among the thresholds keeping precision >= MIN_P is taken, clipped to [LO, HI].
Classes without enough real val keep a default: DEFAULT_REAL if the class has real training images,
DEFAULT_SYN if it is synthetic-only (those fire on look-alike logos in real streets, so stricter).

Writes thresholds.json {class name: threshold} + thresholds_report.csv
usage: python calibrate_thresholds.py [weights] [width]
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

ROOT = Path(__file__).parent
WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "runs" / "jp_signs_v2" / "weights" / "best.pt")
WIDTH = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
VAL = ROOT / "jp_eval_full" / "all"
MIN_GT = 5
LO, HI = 0.25, 0.50
MIN_P = 0.5
DEFAULT_REAL, DEFAULT_SYN = 0.50, 0.65
BETA = 1.5


def main():
    m = YOLO(WEIGHTS)
    names = m.names
    syn_only = set((ROOT / "synthetic_only_classes.txt").read_text().splitlines())
    preds, n_gt = defaultdict(list), defaultdict(int)      # cls -> [(conf, is_tp)]
    imgs = sorted((VAL / "images").glob("*.jpg"))
    for n, p in enumerate(imgs):
        im = cv2.imread(str(p))
        H, W = im.shape[:2]
        s = WIDTH / W
        im = cv2.resize(im, (WIDTH, int(round(H * s))), interpolation=cv2.INTER_AREA)
        h, w = im.shape[:2]
        gt = [list(map(float, l.split())) for l in (VAL / "labels" / (p.stem + ".txt")).read_text().split("\n") if l.strip()]
        gcls = [int(g[0]) for g in gt]
        gbox = torch.tensor([[(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]
                             for _, x, y, bw, bh in gt]).reshape(-1, 4)
        for c in gcls:
            n_gt[c] += 1
        r = m.predict(im, imgsz=int(np.ceil(w / 32) * 32), conf=0.05, verbose=False)[0]
        pb, pc, pk = r.boxes.xyxy.cpu(), r.boxes.conf.cpu().tolist(), r.boxes.cls.cpu().int().tolist()
        used = set()
        for i in np.argsort(pc)[::-1]:
            k, tp = pk[i], False
            if len(gbox):
                iou = torchvision.ops.box_iou(pb[[i]], gbox)[0]
                for j in torch.argsort(iou, descending=True).tolist():
                    if iou[j] < 0.5:
                        break
                    if gcls[j] == k and j not in used:
                        used.add(j); tp = True
                        break
            preds[k].append((pc[i], tp))
        if n % 100 == 0:
            print(f"{n}/{len(imgs)}", flush=True)

    thr, rows = {}, []
    for k, name in names.items():
        g = n_gt.get(k, 0)
        if g < MIN_GT:
            d = DEFAULT_SYN if name in syn_only else DEFAULT_REAL
            thr[name] = d
            rows.append([name, g, d, "", "", "default (synthetic-only)" if name in syn_only else "default (little real val)"])
            continue
        ps = sorted(preds[k], key=lambda t: -t[0])
        best = (-1, HI, 0, 0)
        tp = fp = 0
        for c, is_tp in ps:
            tp += is_tp; fp += not is_tp
            P, R = tp / (tp + fp), tp / g
            f = (1 + BETA ** 2) * P * R / max(1e-9, BETA ** 2 * P + R)
            if P >= MIN_P and f > best[0]:
                best = (f, c, P, R)
        t = float(np.clip(best[1], LO, HI))
        thr[name] = round(t, 3)
        rows.append([name, g, round(t, 3), round(best[2], 3), round(best[3], 3), ""])
    (ROOT / os.environ.get("THRESH_OUT", "thresholds.json")).write_text(json.dumps(thr, indent=1, ensure_ascii=False))
    with open(ROOT / "thresholds_report.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["class", "val_labels", "threshold", "precision_at_thr", "recall_at_thr", "note"])
        wr.writerows(rows)
    cal = [r for r in rows if not r[5]]
    print(f"calibrated {len(cal)} classes, {len(rows) - len(cal)} on defaults {DEFAULT_REAL}/{DEFAULT_SYN}")
    for r in sorted(cal, key=lambda r: r[2]):
        print(f"  {r[2]:.2f}  P {r[3]:.2f} R {r[4]:.2f}  n={r[1]:4d}  {r[0]}")


if __name__ == "__main__":
    main()
