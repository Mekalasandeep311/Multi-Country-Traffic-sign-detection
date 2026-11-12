"""Round 2b: label the OTHER signs in real frames (run after relabel_round2.py).

Each class folder labels only its own sign, so a frame that also shows a sign whose folder does not
contain the frame has that sign unlabeled. In training such a sign is taught as background, and in
the full-frame evaluation a correct detection of it counts as a false positive.

For every unique real frame (by file name; raw names are unique) the union of all folder labels is
taken; the v2 model runs at native resolution and every box with conf >= ACCEPT, aspect within
ASPECT_TOL of the class's median and IoU < 0.3 with every labeled box is added as an "other" label.
The per-folder .txt files in Merged / standard_japan are NOT changed (one class per folder stays);
the extra labels live in autobox/round2_other.csv and are merged per frame by make_train_split.py.

Outputs: autobox/round2_other.csv (file,class_id,class,conf,xc,yc,w,h normalized),
         autobox/review_round2_other/<class>.jpg (up to 60 crops per class)
usage: python relabel_other.py
"""
import csv
import json
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
WEIGHTS = ROOT / "runs" / "jp_signs_v2" / "weights" / "best.pt"
ACCEPT = 0.70
ASPECT_TOL = 1.5
OVERLAP = 0.3


def main():
    folders = yaml.safe_load((ROOT / "class_map.yaml").read_text())["folders"]
    names = dict(enumerate(l for l in (ROOT / "standard_japan" / "classes.txt").read_text().split("\n") if l))
    labels, path_of, aspects = defaultdict(list), {}, defaultdict(list)
    for folder, cid in folders.items():
        if not (AUTO / f"{folder}.json").exists():         # class without real images
            continue
        res = json.loads((AUTO / f"{folder}.json").read_text())
        for f, r in res.items():
            path_of.setdefault(f, DATA / folder / f)
            if r["status"] == "accepted":
                labels[f] += [(cid, b) for b in r["boxes"]]
                aspects[cid] += [(b[2] - b[0]) / max(1e-3, b[3] - b[1]) for b in r["boxes"]]
    med = {c: float(np.median(a)) for c, a in aspects.items() if len(a) >= 5}
    print(f"{len(path_of)} unique frames, {sum(map(len, labels.values()))} folder labels", flush=True)

    model = YOLO(str(WEIGHTS))
    out_dir = AUTO / "review_round2_other"
    out_dir.mkdir(exist_ok=True)
    rows, tiles = [], defaultdict(list)
    for n, (f, p) in enumerate(sorted(path_of.items())):
        im = cv2.imread(str(p))
        H, W = im.shape[:2]
        r = model.predict(im, imgsz=int(np.ceil(max(H, W) / 32) * 32), conf=ACCEPT, verbose=False)[0]
        if not len(r.boxes):
            continue
        b, c, k = r.boxes.xyxy.cpu(), r.boxes.conf.cpu(), r.boxes.cls.cpu().int()
        have = torch.tensor([bb for _, bb in labels[f]], dtype=torch.float32).reshape(-1, 4)
        for i in range(len(b)):
            cid = int(k[i])
            w_, h_ = float(b[i][2] - b[i][0]), float(b[i][3] - b[i][1])
            if cid in med and not (1 / ASPECT_TOL <= (w_ / max(h_, 1e-3)) / med[cid] <= ASPECT_TOL):
                continue
            if len(have) and torchvision.ops.box_iou(b[[i]], have).max() >= OVERLAP:
                continue
            x0, y0, x1, y1 = b[i].tolist()
            rows.append([f, cid, names[cid], round(float(c[i]), 3), round((x0 + x1) / 2 / W, 6),
                         round((y0 + y1) / 2 / H, 6), round((x1 - x0) / W, 6), round((y1 - y0) / H, 6)])
            if len(tiles[cid]) < 60:
                m = 0.8 * max(x1 - x0, y1 - y0) + 8
                ox, oy = int(max(0, x0 - m)), int(max(0, y0 - m))
                crop = im[oy:int(y1 + m), ox:int(x1 + m)].copy()
                cv2.rectangle(crop, (int(x0) - ox, int(y0) - oy), (int(x1) - ox, int(y1) - oy), (0, 255, 0), 1)
                crop = cv2.resize(crop, (128, 128))
                cv2.putText(crop, f"{float(c[i]):.2f}", (3, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                tiles[cid].append(crop)
        if n % 1000 == 0:
            print(f"{n}/{len(path_of)} frames, {len(rows)} other labels", flush=True)
    for cid, t in tiles.items():
        t += [np.zeros_like(t[0])] * (-len(t) % 10)
        cv2.imwrite(str(out_dir / f"{names[cid]}.jpg"), np.vstack([np.hstack(t[i:i + 10]) for i in range(0, len(t), 10)]))
    with open(AUTO / "round2_other.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "class_id", "class", "conf", "xc", "yc", "w", "h"])
        w.writerows(rows)
    per = defaultdict(int)
    for r in rows:
        per[r[2]] += 1
    print(f"TOTAL other labels: {len(rows)} in {len({r[0] for r in rows})} frames")
    for k_, v in sorted(per.items(), key=lambda x: -x[1]):
        print(f"  {k_}: {v}")


if __name__ == "__main__":
    main()
