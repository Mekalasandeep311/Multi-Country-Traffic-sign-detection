"""Round-2 relabeling: label the uncertain / rejected real frames with the trained v2 model.

For every frame of folder class C that has no label yet (status uncertain or rejected in
autobox/<folder>.json), run runs/sg_signs_v2/weights/best.pt at native resolution and accept the
box(es) of class C with conf >= ACCEPT, unless a box of a different class overlaps it with a
higher score. Accepted frames get their YOLO .txt next to the image (only class C boxes, as in
round 1) and status "accepted" (reason "round2 v2 model") in the json.

Outputs: autobox/round2_report.csv, autobox/review_round2/<folder>.jpg
usage: python relabel_round2.py
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).parent
DATA = ROOT / "Traffic Signs Singapore"
AUTO = ROOT / "autobox"
WEIGHTS = ROOT / "runs" / "sg_signs_v2" / "weights" / "best.pt"
ACCEPT = 0.70
ASPECT_TOL = 1.5          # box w/h must be within this factor of the class's round-1 median
SKIP = {"Work Access", "Works Ahead"}   # look-alike pair: v2 mixes them up


def restore_round1():
    """Undo an earlier round-2 run: restore round-1 jsons and delete round-2 label files."""
    bak = ROOT / "autobox_backup_round1"
    for jf in AUTO.glob("*.json"):
        cur = json.loads(jf.read_text())
        old = json.loads((bak / jf.name).read_text()) if (bak / jf.name).exists() else cur
        for f, r in cur.items():
            if r.get("reason") == "round2 v2 model":
                (DATA / jf.stem / (Path(f).stem + ".txt")).unlink(missing_ok=True)
        jf.write_text(json.dumps(old, indent=1))


def main():
    folders = yaml.safe_load((ROOT / "class_map.yaml").read_text())["folders"]
    restore_round1()
    model = YOLO(str(WEIGHTS))
    (AUTO / "review_round2").mkdir(exist_ok=True)
    rows = []
    for folder, cid in folders.items():
        jf = AUTO / f"{folder}.json"
        res = json.loads(jf.read_text())
        todo = [] if folder in SKIP else [f for f, r in res.items() if r["status"] != "accepted"]
        asp = [(b[2] - b[0]) / max(1e-3, b[3] - b[1]) for r in res.values() if r["status"] == "accepted" for b in r["boxes"]]
        med_asp = float(np.median(asp)) if asp else None
        new, tiles = 0, []
        for f in todo:
            im = cv2.imread(str(DATA / folder / f))
            H, W = im.shape[:2]
            r = model.predict(im, imgsz=int(np.ceil(max(H, W) / 32) * 32), conf=0.25, verbose=False)[0]
            b, c, k = r.boxes.xyxy.cpu(), r.boxes.conf.cpu(), r.boxes.cls.cpu().int()
            mine = [(i, float(c[i])) for i in range(len(b)) if k[i] == cid and c[i] >= ACCEPT]
            others = [i for i in range(len(b)) if k[i] != cid]
            keep = []
            for i, s in mine:
                w_, h_ = float(b[i][2] - b[i][0]), float(b[i][3] - b[i][1])
                if med_asp and not (1 / ASPECT_TOL <= (w_ / max(h_, 1e-3)) / med_asp <= ASPECT_TOL):
                    continue
                if others:
                    iou = torchvision.ops.box_iou(b[[i]], b[others])[0]
                    if any(iou[j] > 0.3 and c[others[j]] > s for j in range(len(others))):
                        continue
                keep.append(i)
            if not keep:
                continue
            boxes = [b[i].tolist() for i in keep]
            (DATA / folder / (Path(f).stem + ".txt")).write_text("".join(
                f"{cid} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n"
                for x0, y0, x1, y1 in boxes))
            res[f].update({"status": "accepted", "reason": "round2 v2 model",
                           "score": max(float(c[i]) for i in keep), "boxes": [[round(v, 1) for v in bb] for bb in boxes]})
            new += 1
            if len(tiles) < 60:
                x0, y0, x1, y1 = boxes[0]
                m = 0.8 * max(x1 - x0, y1 - y0) + 8
                ox, oy = int(max(0, x0 - m)), int(max(0, y0 - m))
                crop = im[oy:int(y1 + m), ox:int(x1 + m)].copy()
                cv2.rectangle(crop, (int(x0) - ox, int(y0) - oy), (int(x1) - ox, int(y1) - oy), (0, 255, 0), 1)
                crop = cv2.resize(crop, (128, 128))
                cv2.putText(crop, f"{res[f]['score']:.2f}", (3, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                tiles.append(crop)
        jf.write_text(json.dumps(res, indent=1))
        if tiles:
            tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 10)
            cv2.imwrite(str(AUTO / "review_round2" / f"{folder}.jpg"),
                        np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]))
        total_acc = sum(r["status"] == "accepted" for r in res.values())
        rows.append([folder, cid, len(res), len(todo), new, total_acc])
        print(f"{folder}: {new}/{len(todo)} newly labeled (now {total_acc}/{len(res)})", flush=True)
    with open(AUTO / "round2_report.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "class_id", "images", "unlabeled_before", "newly_labeled", "labeled_now"])
        w.writerows(rows)
    print("TOTAL newly labeled:", sum(r[4] for r in rows), "of", sum(r[3] for r in rows),
          "| labeled now:", sum(r[5] for r in rows), "of", sum(r[2] for r in rows))


if __name__ == "__main__":
    main()
