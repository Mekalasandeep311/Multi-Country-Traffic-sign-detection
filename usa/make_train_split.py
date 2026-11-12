"""Train/val/test split + 640 training tiles from standard_usa (USA, 209 classes).

Sources in standard_usa/<class>/:
  real_<source>__<frame>.jpg   US drive frame (auto-boxed)          -> train / real val
  real_mtsd_train__<key>.jpg   Mapillary MTSD train image          -> train only
  real_mtsd_val__<key>.jpg     Mapillary MTSD val image            -> held-out TEST, never trained on
  syn_*.jpg                    synthetic sign swap                 -> train (+ small val share, see below)

Val is REAL US drive frames only: per class, a contiguous block (by source + frame number) of ~VAL_SHARE of
its drive frames. Train drive frames of the same video within NEIGHBOR frame numbers of any val frame are
removed (the same sign, seconds apart, would otherwise leak into val). Classes with < 3 drive frames get a
small synthetic val share instead so every class is still measured (reported separately).

The same frame can sit in several class folders (one per sign it shows); every copy gets the UNION of all its
labels, and a frame in val/test is removed from train entirely.

Uncertain boxes (autobox/uncertain.json: sign-like boxes the auto-labeller would not name) are greyed out
(114, like the YOLO letterbox) in every drive-frame tile, so an unlabelled real sign is never learned as
background. Labelled boxes are never greyed (uncertain boxes can overlap them). MTSD images were greyed
when copied (mtsd_to_real.py).

Tiles: each image -> one 640x640 crop at native resolution containing its labelled sign(s) at a random
offset (+ TILE_NEG_SHARE extra sign-free crops as negatives). A sign larger than the tile: the image is
downscaled first.

Output <OUT>/
  images|labels/train, val          640 tiles
  images|labels/val_full            full drive val frames (greyed, union labels)  -> data_val_full.yaml
  images|labels/test_mtsd           full MTSD-val images (union labels)           -> data.yaml "test"
  data.yaml, data_val_full.yaml, split_report.csv

usage: python make_train_split.py [subset_share]   (subset_share < 1 for the LR check: train subset only,
                                                    same val, no test/val_full copies)
"""
import csv
import json
import os
import random
import shutil
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).parent
BAL = ROOT / "standard_usa"
SUBSET = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
OUT = ROOT / os.environ.get("SPLIT_OUT", "us_train" if SUBSET >= 1 else f"us_train_{int(SUBSET * 100)}pct")
UNCERTAIN = ROOT / "autobox" / "uncertain.json"
NEIGHBOR = int(os.environ.get("NEIGHBOR", 60))   # frame numbers (drive frames are every 5th video frame)
# OTHER_DIR (env): folder holding one extra class folder, e.g. other_sign/ from make_other_signs.py; the class
# is appended after classes.txt (id 209). Val/test of classes 0-208 stay identical to the split without it.
OTHER_DIR = Path(os.environ["OTHER_DIR"]) if os.environ.get("OTHER_DIR") else None
OTHER_NAME = "Other sign (not in list)"
# EXTRA_REAL (env): folder of hand-checked full frames (*.jpg + *.txt, unsure signs already greyed), e.g.
# clip2_labeled/ from mine_clip2.py; added to TRAIN only (never val/test)
EXTRA_REAL = Path(os.environ["EXTRA_REAL"]) if os.environ.get("EXTRA_REAL") else None
VAL_SHARE = 0.10
SYN_VAL_SHARE = 0.05
TILE = 640
TILE_NEG_SHARE = 0.10
GREY = 114
DUP_IOU = 0.3            # uncertain box overlapping a label this much = duplicate proposal of that sign


def kind(p):
    if EXTRA_REAL and p.parent == EXTRA_REAL:
        return "extra"
    if p.name.startswith("syn_"):
        return "syn"
    if p.name.startswith("real_mtsd_val__"):
        return "mtsd_val"
    if p.name.startswith("real_mtsd_"):
        return "mtsd"
    return "drive"


def read_labels(p):
    t = p.with_suffix(".txt")
    return [list(map(float, l.split())) for l in t.read_text().split("\n") if l.strip()] if t.exists() else []


def frame_key(p):
    """Same raw frame in different class folders (hard links): same original name and size."""
    return p.name[len("real_"):], p.stat().st_size


def src_num(p):
    """Drive frame -> (source video, frame number)."""
    src, rest = p.stem[len("real_"):].split("__", 1)
    d = "".join(ch for ch in rest if ch.isdigit())
    return src, int(d) if d else -1


def unc_key(p):
    """standard_usa name real_<source>__<frame>.jpg -> uncertain.json key <source>/<frame>.jpg"""
    return p.name[len("real_"):].replace("__", "/", 1)


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def load_greyed(img_path, labels, grey):
    """Image with uncertain boxes greyed, label boxes untouched; returns (image, label boxes in px)."""
    im = cv2.imread(str(img_path))
    H, W = im.shape[:2]
    boxes = np.array([[(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]
                      for _, x, y, w, h in labels]).reshape(-1, 4)
    if grey:
        m = np.zeros((H, W), bool)
        for g in grey:
            if any(iou(g, b) >= DUP_IOU for b in boxes):
                continue
            m[max(0, int(g[1])):int(np.ceil(g[3])), max(0, int(g[0])):int(np.ceil(g[2]))] = True
        for b in boxes:
            m[max(0, int(b[1])):int(np.ceil(b[3])), max(0, int(b[0])):int(np.ceil(b[2]))] = False
        im[m] = GREY
    return im, boxes


def yolo_lines(cls, boxes, x0, y0, tw, th):
    lines = []
    for c, (a0, b0, a1, b1) in zip(cls, boxes):
        a0, a1 = np.clip([a0 - x0, a1 - x0], 0, tw)
        b0, b1 = np.clip([b0 - y0, b1 - y0], 0, th)
        if a1 - a0 > 2 and b1 - b0 > 2:
            lines.append(f"{c} {(a0 + a1) / 2 / tw:.6f} {(b0 + b1) / 2 / th:.6f} {(a1 - a0) / tw:.6f} {(b1 - b0) / th:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def make_tile(job):
    img_path, out_stem, seed, negative, labels, grey = job
    rng = np.random.default_rng(seed)
    im, boxes = load_greyed(img_path, labels, grey)
    H, W = im.shape[:2]
    cls = [int(l[0]) for l in labels]
    if negative:                                   # crop away from the labelled signs
        for _ in range(20):
            x0, y0 = rng.integers(0, max(1, W - TILE)), rng.integers(0, max(1, H - TILE))
            if not len(boxes) or not ((boxes[:, 2] > x0) & (boxes[:, 0] < x0 + TILE) &
                                      (boxes[:, 3] > y0) & (boxes[:, 1] < y0 + TILE)).any():
                cv2.imwrite(f"{out_stem}.jpg", im[y0:y0 + TILE, x0:x0 + TILE])
                Path(f"{out_stem}.txt").write_text("")
                return
        return
    span_w, span_h = boxes[:, 2].max() - boxes[:, 0].min(), boxes[:, 3].max() - boxes[:, 1].min()
    s = min(1.0, 0.9 * TILE / max(span_w, span_h, 1))
    if s < 1:
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        boxes = boxes * s
        H, W = im.shape[:2]
    bx0, by0, bx1, by1 = boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()
    x0 = int(np.clip(rng.uniform(bx1 - TILE, bx0), 0, max(0, W - TILE)))
    y0 = int(np.clip(rng.uniform(by1 - TILE, by0), 0, max(0, H - TILE)))
    tile = im[y0:y0 + TILE, x0:x0 + TILE]
    cv2.imwrite(f"{out_stem}.jpg", tile)
    Path(f"{out_stem}.txt").write_text(yolo_lines(cls, boxes, x0, y0, tile.shape[1], tile.shape[0]))


def make_full(job):
    img_path, out_stem, labels, grey = job
    im, boxes = load_greyed(img_path, labels, grey)
    cv2.imwrite(f"{out_stem}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    Path(f"{out_stem}.txt").write_text(yolo_lines([int(l[0]) for l in labels], boxes, 0, 0, im.shape[1], im.shape[0]))


def main():
    names = [l for l in (BAL / "classes.txt").read_text().split("\n") if l]
    base = {cid: BAL for cid in range(len(names))}
    if OTHER_DIR:
        base[len(names)] = OTHER_DIR
        names.append(OTHER_NAME)
    unc = json.loads(UNCERTAIN.read_text())
    rnd = random.Random(0)
    if OUT.exists():
        shutil.rmtree(OUT)                         # only our own split output lives here
    splits = ("train", "val") if SUBSET < 1 else ("train", "val", "val_full", "test_mtsd")
    for s in splits:
        (OUT / "images" / s).mkdir(parents=True)
        (OUT / "labels" / s).mkdir(parents=True)

    train, val, val_full, test, syn_val_classes, rows = [], [], [], [], [], []
    for cid, name in enumerate(names):
        cdir = base[cid] / name.replace("/", "-").replace(":", "-").strip()
        files = sorted(cdir.glob("*.jpg"))
        by = {k: [p for p in files if kind(p) == k and (k == "syn" or read_labels(p))]
              for k in ("drive", "mtsd", "mtsd_val", "syn")}
        drive = sorted(by["drive"], key=src_num)
        syn = by["syn"]
        test += by["mtsd_val"]
        if len(drive) >= 3:
            n_val = max(1, int(round(VAL_SHARE * len(drive))))
            start = rnd.randint(0, len(drive) - n_val)
            v = drive[start:start + n_val]
            vt = [p for p in drive if p not in set(v)] + by["mtsd"] + syn
            val += v; val_full += v
        elif syn:
            n_val = max(2, int(round(SYN_VAL_SHARE * len(syn))))
            v = syn[-n_val:]
            vt = drive + by["mtsd"] + syn[:-n_val]
            val += v; syn_val_classes.append(name)
        else:
            v, vt = [], drive + by["mtsd"] + syn
        train += vt
        rows.append([cid, name, len(drive), len(by["mtsd"]), len(by["mtsd_val"]), len(syn),
                     sum(1 for p in v if kind(p) == "drive"), sum(1 for p in v if kind(p) == "syn")])

    # union of labels per real frame over all class folders (kept + extra copies)
    union = {}
    for p in list(BAL.glob("*/real_*.jpg")) + list(BAL.glob("*/extra/real_*.jpg")):
        lines = {" ".join(f"{v:g}" if i == 0 else f"{v:.6f}" for i, v in enumerate(l)) for l in read_labels(p)}
        union.setdefault(frame_key(p), set()).update(lines)

    # train: no copy of a val/test frame, no drive frame next to a val frame of the same video
    hold = {frame_key(p) for p in val_full + test}
    vnum = {}
    for p in val_full:
        s, n = src_num(p)
        vnum.setdefault(s, []).append(n)
    vnum = {s: np.array(sorted(v)) for s, v in vnum.items()}

    def near_val(p):
        s, n = src_num(p)
        a = vnum.get(s)
        if a is None or n < 0:
            return False
        i = np.searchsorted(a, n)
        return any(0 <= j < len(a) and abs(a[j] - n) <= NEIGHBOR for j in (i - 1, i))

    n0 = len(train)
    train = [p for p in train if kind(p) == "syn" or frame_key(p) not in hold]
    n1 = len(train)
    train = [p for p in train if kind(p) != "drive" or not near_val(p)]
    print(f"train copies removed: {n0 - n1} (frame in val/test), {n1 - len(train)} (within {NEIGHBOR} frames "
          f"of a val frame)", flush=True)
    if EXTRA_REAL:
        extra = [p for p in sorted(EXTRA_REAL.glob("*.jpg")) if read_labels(p)]
        train += extra
        print(f"hand-checked extra real frames added to train: {len(extra)}", flush=True)
    if SUBSET < 1:
        train = rnd.sample(train, int(SUBSET * len(train)))

    def labels_of(p):
        if kind(p) in ("syn", "extra"):
            return read_labels(p)
        return [list(map(float, l.split())) for l in sorted(union[frame_key(p)])]

    def grey_of(p):
        return unc.get(unc_key(p), []) if kind(p) == "drive" else []

    n_unc = sum(1 for p in train + val if kind(p) == "drive" and unc_key(p) in unc)
    n_drv = sum(1 for p in train + val if kind(p) == "drive")
    print(f"drive tiles with uncertain boxes to grey: {n_unc} of {n_drv}", flush=True)

    jobs = []
    for split, files in (("train", train), ("val", val)):
        for i, p in enumerate(files):
            stem = f"{p.parent.name.replace(' ', '_')}__{p.stem}"
            lab, g = labels_of(p), grey_of(p)
            jobs.append((p, str(OUT / "images" / split / stem), i, False, lab, g))
            if split == "train" and kind(p) != "syn" and rnd.random() < 2 * TILE_NEG_SHARE:
                jobs.append((p, str(OUT / "images" / split / (stem + "_neg")), i + 10 ** 7, True, lab, g))
            elif split == "train" and kind(p) == "syn" and rnd.random() < TILE_NEG_SHARE / 2:
                jobs.append((p, str(OUT / "images" / split / (stem + "_neg")), i + 10 ** 7, True, lab, g))
    full_jobs = []
    if SUBSET >= 1:
        seen = set()
        for split, files in (("val_full", val_full), ("test_mtsd", test)):
            for p in files:
                if frame_key(p) in seen:           # one copy per frame (labels are the union anyway)
                    continue
                seen.add(frame_key(p))
                full_jobs.append((p, str(OUT / "images" / split / p.stem), labels_of(p), grey_of(p)))
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as pool:
        pool.map(make_tile, jobs, chunksize=64)
        pool.map(make_full, full_jobs, chunksize=16)
    for s in splits:                               # move .txt files into labels/
        for t in (OUT / "images" / s).glob("*.txt"):
            t.rename(OUT / "labels" / s / t.name)

    nm = dict(enumerate(names))
    data = {"path": str(OUT), "train": "images/train", "val": "images/val", "names": nm}
    if SUBSET >= 1:
        data["test"] = "images/test_mtsd"
    (OUT / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    if SUBSET >= 1:
        (OUT / "data_val_full.yaml").write_text(yaml.safe_dump(
            {"path": str(OUT), "train": "images/train", "val": "images/val_full", "names": nm},
            sort_keys=False, allow_unicode=True))
    with open(OUT / "split_report.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class_id", "class", "drive", "mtsd_train", "mtsd_test", "synthetic", "val_drive", "val_syn"])
        w.writerows(rows)
    c = Counter(kind(p) for p in train)
    print(f"train images {len(train)} (drive {c['drive']}, MTSD {c['mtsd']}, synthetic {c['syn']}, "
          f"extra real {c['extra']}), "
          f"tiles incl. negatives {len(jobs) - len(val)}; val {len(val)} (real {len(val_full)}, "
          f"{len(syn_val_classes)} classes with synthetic-only val); val_full frames "
          f"{sum(1 for j in full_jobs if '/val_full/' in j[1])}, MTSD test images "
          f"{sum(1 for j in full_jobs if '/test_mtsd/' in j[1])}")


if __name__ == "__main__":
    main()
