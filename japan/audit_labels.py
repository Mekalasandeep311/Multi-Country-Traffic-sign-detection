"""Audit the real labels (autobox + round 2) with the v2 model.

For every accepted label box of folder class C the model's best prediction overlapping it
(IoU >= 0.5, any class, conf >= 0.05) is looked up at native resolution:
  agree     - top overlapping prediction is class C
  other     - top overlapping prediction is another class (label may box the wrong sign)
  none      - nothing found there (label may box no sign at all)
Outputs autobox/audit.csv (one row per label), autobox/audit_summary.csv (per class rates) and
autobox/audit_sheets/<class>.jpg: up to 40 doubtful crops (other/none first, model's guess printed)
followed by 10 agreeing crops for comparison.
usage: python audit_labels.py [weights]
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).parent
DATA = ROOT / "classified Data japan" / "Merged"
AUTO = ROOT / "autobox"
WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "runs" / "jp_signs_v2" / "weights" / "best.pt")


def crop(im, b, text, ok):
    x0, y0, x1, y1 = map(int, b)
    m = int(0.9 * max(x1 - x0, y1 - y0)) + 8
    ox, oy = max(0, x0 - m), max(0, y0 - m)
    c = im[oy:y1 + m, ox:x1 + m].copy()
    cv2.rectangle(c, (x0 - ox, y0 - oy), (x1 - ox, y1 - oy), (0, 255, 0) if ok else (0, 0, 255), 2)
    c = cv2.resize(c, (160, 160))
    cv2.rectangle(c, (0, 0), (160, 16), (0, 0, 0), -1)
    cv2.putText(c, text[:30], (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)
    return c


def main():
    folders = yaml.safe_load((ROOT / "class_map.yaml").read_text())["folders"]
    m = YOLO(WEIGHTS)
    names = m.names
    rows, by_class = [], defaultdict(list)
    for folder, cid in folders.items():
        jf = AUTO / f"{folder}.json"
        if not jf.exists():
            continue
        res = json.loads(jf.read_text())
        for f, r in res.items():
            if r["status"] != "accepted":
                continue
            im = cv2.imread(str(DATA / folder / f))
            H, W = im.shape[:2]
            p = m.predict(im, imgsz=int(np.ceil(max(H, W) / 32) * 32), conf=0.05, verbose=False)[0]
            pb, pc, pk = p.boxes.xyxy.cpu(), p.boxes.conf.cpu(), p.boxes.cls.cpu().int()
            for b in r["boxes"]:
                verdict, guess, gconf = "none", "", 0.0
                if len(pb):
                    iou = torchvision.ops.box_iou(torch.tensor([b], dtype=torch.float32), pb)[0]
                    sel = (iou >= 0.5).nonzero().flatten().tolist()
                    if sel:
                        j = max(sel, key=lambda j: float(pc[j]))
                        guess, gconf = names[int(pk[j])], float(pc[j])
                        own = [j for j in sel if int(pk[j]) == cid]
                        verdict = "agree" if int(pk[j]) == cid else "other"
                        if verdict == "other" and own and float(pc[own[0]]) >= 0.5 * gconf:
                            verdict = "agree"          # own class is a close second
                row = [folder, f, cid, [round(v, 1) for v in b], verdict, guess, round(gconf, 3), r.get("reason", "")]
                rows.append(row)
                text = f"{verdict}: {guess} {gconf:.2f}" if verdict != "agree" else f"ok {gconf:.2f}"
                by_class[folder].append((row, crop(im, b, text, verdict == "agree")))
        print(folder, flush=True)

    with open(AUTO / "audit.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["folder", "file", "class_id", "box", "verdict", "model_guess", "model_conf", "label_source"])
        w.writerows(rows)
    out = AUTO / "audit_sheets"
    out.mkdir(exist_ok=True)
    summ = []
    for folder, items in by_class.items():
        n = len(items)
        cnt = {v: sum(1 for r, _ in items if r[4] == v) for v in ("agree", "other", "none")}
        summ.append([folder, n, cnt["agree"], cnt["other"], cnt["none"], round(cnt["agree"] / n, 3)])
        bad = [it for it in items if it[0][4] != "agree"]
        good = [it for it in items if it[0][4] == "agree"]
        rng = np.random.default_rng(0)
        pick = [bad[i] for i in sorted(rng.choice(len(bad), min(40, len(bad)), replace=False))] if bad else []
        pick += [good[i] for i in sorted(rng.choice(len(good), min(10, len(good)), replace=False))] if good else []
        tiles = [t for _, t in pick]
        if not tiles:
            continue
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 10)
        cv2.imwrite(str(out / f"{folder}.jpg"), np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]))
    summ.sort(key=lambda r: r[5])
    with open(AUTO / "audit_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["folder", "labels", "agree", "other", "none", "agree_rate"])
        w.writerows(summ)
    for r in summ:
        print(f"{r[5]:.2f}  n={r[1]:5d} other={r[3]:4d} none={r[4]:4d}  {r[0]}")


if __name__ == "__main__":
    main()
