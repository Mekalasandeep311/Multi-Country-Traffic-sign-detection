"""Copy only the MTSD images our classes are short of into mtsd_by_class/ (user request 2026-10-31).

For every class in mtsd_map.MAPPING_OK: need = TARGET - real drive frames (autobox/report.csv).
Picks up to `need` MTSD images (train+val, fully annotated, not panoramas) that show the class at a usable
size, extracts just those files from the zips, and writes
  mtsd_by_class/_images/mtsd_<split>__<key>.jpg      resized to MAX_SIDE; every sign that is NOT a mapped,
                                                     clear label is greyed out (so it is never learned as
                                                     background)
  mtsd_by_class/<class>/mtsd_<split>__<key>.jpg/.txt hard link + YOLO labels of all mapped signs, own class first
  mtsd_by_class/report.csv, LICENSE.txt (MTSD is CC BY-NC-SA: keep attribution)
The split is kept in the name so MTSD-val images can be held out for testing later.

usage: python mtsd_to_real.py
"""
import csv
import os
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from mtsd_map import ANN, MAPPING_OK, MT, load_ann

ROOT = Path(__file__).parent
OUT = ROOT / "mtsd_by_class"
TARGET = 400
MAX_SIDE = 1920          # our drive frames are 1280x720; MTSD photos go up to 4000+ px
MIN_SIDE_SELECT = 20     # px (original) the class's sign must have for the image to be picked
MIN_SIDE_LABEL = 8       # px (after resize) for a mapped sign to be labelled; smaller ones are greyed
ZIPS = [MT / f"mtsd_v2_fully_annotated_images.{s}.zip" for s in ("train.0", "train.1", "train.2", "val")]


def safe(name):
    return name.replace("/", "-").replace(":", "-").strip()


def clear(o):
    p = o["properties"]
    return not (p.get("dummy") or p.get("ambiguous") or p.get("out-of-frame"))


def main():
    names = yaml.safe_load((ROOT / "class_map.yaml").read_text())["names"]
    drive = {int(r["class_id"]): int(r["frames"]) for r in csv.DictReader(open(ROOT / "autobox" / "report.csv"))}
    split = {}
    for s in ("train", "val"):
        for k in open(ANN / "splits" / f"{s}.txt").read().split():
            split[k] = s

    cand = defaultdict(list)                 # cid -> [(key, biggest side of that class's sign)]
    anns = {}
    for k, a in load_ann(set(split)):
        if a.get("ispano"):
            continue
        best = defaultdict(float)
        for o in a["objects"]:
            cid = MAPPING_OK.get(o["label"])
            if cid is None or not clear(o) or o["properties"].get("occluded"):
                continue
            b = o["bbox"]
            best[cid] = max(best[cid], min(b["xmax"] - b["xmin"], b["ymax"] - b["ymin"]))
        for cid, s in best.items():
            if s >= MIN_SIDE_SELECT:
                cand[cid].append((k, s))
                anns[k] = a

    picked = defaultdict(list)               # cid -> keys
    for cid, lst in cand.items():
        need = max(0, TARGET - drive.get(cid, 0))
        rng = random.Random(cid)
        rng.shuffle(lst)
        lst.sort(key=lambda t: t[1] < 32)    # larger signs first, random within
        picked[cid] = [k for k, _ in lst[:need]]
    keys = sorted({k for v in picked.values() for k in v})
    print(f"{len(keys)} MTSD images for {sum(1 for v in picked.values() if v)} classes", flush=True)

    # extract only the needed files
    tmp = MT / "extract"
    tmp.mkdir(exist_ok=True)
    want = {f"images/{k}.jpg" for k in keys}
    for z in ZIPS:
        if not z.exists():
            print("missing", z); continue
        members = [m for m in subprocess.run(["unzip", "-Z1", str(z)], capture_output=True, text=True).stdout.split()
                   if m in want]
        for i in range(0, len(members), 500):
            subprocess.run(["unzip", "-q", "-o", str(z), *members[i:i + 500], "-d", str(tmp)], check=True)
        print(f"{z.name}: {len(members)} extracted", flush=True)

    if OUT.exists():
        shutil.rmtree(OUT)                   # only our own output lives here
    (OUT / "_images").mkdir(parents=True)
    shutil.copy2(ANN.parent / "LICENSE.txt", OUT / "LICENSE.txt")
    labels, missing = {}, 0
    for k in keys:
        src = tmp / "images" / f"{k}.jpg"
        if not src.exists():
            missing += 1
            continue
        im = cv2.imread(str(src))
        H0, W0 = im.shape[:2]
        f = min(1.0, MAX_SIDE / max(H0, W0))
        if f < 1:
            im = cv2.resize(im, (round(W0 * f), round(H0 * f)), interpolation=cv2.INTER_AREA)
        H, W = im.shape[:2]
        lab, grey = [], np.zeros((H, W), bool)
        for o in anns[k]["objects"]:
            b = o["bbox"]
            x0, y0, x1, y1 = (max(0.0, b["xmin"] * f), max(0.0, b["ymin"] * f),
                              min(W - 1.0, b["xmax"] * f), min(H - 1.0, b["ymax"] * f))
            cid = MAPPING_OK.get(o["label"])
            if cid is not None and clear(o) and min(x1 - x0, y1 - y0) >= MIN_SIDE_LABEL:
                lab.append((cid, x0, y0, x1, y1))
            elif x1 > x0 and y1 > y0:        # unmapped / unclear sign: grey it out
                grey[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = True
        for _, x0, y0, x1, y1 in lab:        # never grey over a labelled sign (MTSD boxes can overlap)
            grey[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = False
        im[grey] = 114                       # same grey as the YOLO letterbox
        name = f"mtsd_{split[k]}__{k}"
        cv2.imwrite(str(OUT / "_images" / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        labels[k] = (name, W, H, lab)
    print(f"written {len(labels)} images ({missing} not found in the zips)", flush=True)

    rows = []
    for cid in range(len(names)):
        n = 0
        d = OUT / safe(names[cid])
        for k in picked.get(cid, []):
            if k not in labels:
                continue
            name, W, H, lab = labels[k]
            own = [l for l in lab if l[0] == cid]
            if not own:                      # its sign fell under MIN_SIDE_LABEL after resizing
                continue
            d.mkdir(exist_ok=True)
            os.link(OUT / "_images" / f"{name}.jpg", d / f"{name}.jpg")
            (d / f"{name}.txt").write_text("".join(
                f"{c} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n"
                for c, x0, y0, x1, y1 in own + [l for l in lab if l[0] != cid]))
            n += 1
        if n or drive.get(cid, 0):
            rows.append([cid, names[cid], drive.get(cid, 0), len(cand.get(cid, [])), n, drive.get(cid, 0) + n])
    with open(OUT / "report.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class_id", "class", "drive_frames", "mtsd_available", "mtsd_copied", "real_total"])
        w.writerows(rows)
    shutil.rmtree(tmp)
    print(f"classes with MTSD images: {sum(1 for r in rows if r[4])}; MTSD images: {len(labels)}")


if __name__ == "__main__":
    main()
