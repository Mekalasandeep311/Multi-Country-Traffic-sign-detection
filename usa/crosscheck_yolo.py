"""Cross-check accepted auto-boxes with the synthetic-trained YOLO.

For every accepted label, take YOLO's per-class scores at that box (anchors with IoU >= 0.5). If YOLO is
confident the box is a DIFFERENT sign (other-class score >= OTHER_MIN and > RATIO x the folder class
score), the image is demoted to "uncertain": its .txt is removed and autobox/<class>.json updated.
Writes autobox/crosscheck.csv (class, file, folder-class score, other class, other score).

usage: python crosscheck_yolo.py
"""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "models"))

import numpy as np
import torch
import torchvision
import yaml
from PIL import Image

from autobox_real import DATA, DEVICE, OUT, YOLO_WEIGHTS

OTHER_MIN = 0.35
RATIO = 1.5


class YoloAll:
    def __init__(self):
        from ultralytics import YOLO
        self.net = YOLO(str(YOLO_WEIGHTS)).model.to(DEVICE).eval().half()
        self.names = self.net.names

    @torch.no_grad()
    def __call__(self, rgb):
        H, W = rgb.shape[:2]
        Hp, Wp = -(-H // 32) * 32, -(-W // 32) * 32
        x = torch.zeros((1, 3, Hp, Wp), device=DEVICE, dtype=torch.half)
        x[0, :, :H, :W] = torch.from_numpy(rgb).to(DEVICE).permute(2, 0, 1).half() / 255
        out = self.net(x)
        pred = (out[0] if isinstance(out, (list, tuple)) else out)[0].float()
        xywh, cls = pred[:4].T, pred[4:].T
        boxes = torch.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                             xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
        return boxes, cls


def main():
    cmap = yaml.safe_load((Path(__file__).parent / "class_map.yaml").read_text())
    folders, names = cmap["folders"], cmap["names"]
    yolo = YoloAll()
    nc = len(yolo.names)
    rows, demoted = [], 0
    for folder, cid in folders.items():
        jf = OUT / f"{folder}.json"
        if not jf.exists():          # empty class folder (synthetic-only class)
            continue
        res = json.loads(jf.read_text())
        changed = False
        for f, r in res.items():
            if r["status"] != "accepted":
                continue
            rgb = np.array(Image.open(DATA / folder / f).convert("RGB"))
            boxes, cls = yolo(rgb)
            b = torch.tensor([r["boxes"][0]], device=DEVICE, dtype=torch.float32)
            m = torchvision.ops.box_iou(b, boxes)[0] >= 0.5
            if not m.any():
                continue
            s = cls[m].max(0).values                                  # per-class max over matching anchors
            own = float(s[cid]) if cid < nc else 0.0
            if cid < nc:
                s[cid] = -1                                           # look at the best OTHER class
            other = int(s.argmax())
            os_ = float(s[other])
            if os_ >= OTHER_MIN and os_ > RATIO * own:
                r["status"], r["reason"] = "uncertain", f"YOLO says {yolo.names[other]} ({os_:.2f})"
                (DATA / folder / (Path(f).stem + ".txt")).unlink(missing_ok=True)
                rows.append([folder, f, round(own, 3), yolo.names[other], round(os_, 3)])
                demoted += 1
                changed = True
        if changed:
            jf.write_text(json.dumps(res, indent=1))
        print(f"{folder}: demoted {sum(1 for x in rows if x[0] == folder)}", flush=True)
    with open(OUT / "crosscheck.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "file", "own_score", "yolo_other_class", "other_score"])
        w.writerows(rows)
    print("total demoted:", demoted)


if __name__ == "__main__":
    main()
