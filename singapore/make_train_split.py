"""Train/val split + 640 training tiles from sg_signs_balanced.

Val is REAL images only: per class, a contiguous block (by file order, so consecutive near-identical
video frames do not leak between train and val) of ~VAL_SHARE of the real images. Classes with <3
real images get no real val; classes with no real images get a small synthetic val share so they
are still measured (reported separately).

Tiles: each image -> one 640x640 crop at native resolution that contains its labeled sign(s) at a
random offset (+ TILE_NEG_SHARE extra sign-free crops from real frames as negatives). A sign larger
than the tile is handled by downscaling that image first.

sg_signs_train/images/{train,val}, labels/{train,val}, data.yaml, val_full.txt (full real frames)

usage: python make_train_split.py [subset_share]   (subset_share < 1 for the LR check)
"""
import os
import random
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).parent
BAL = ROOT / "sg_signs_balanced"
SUBSET = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
OUT = ROOT / ("sg_signs_train" if SUBSET >= 1 else f"sg_signs_train_{int(SUBSET * 100)}pct")
VAL_SHARE = 0.10
SYN_VAL_SHARE = 0.05
TILE = 640
TILE_NEG_SHARE = 0.10


def read_labels(p):
    t = p.with_suffix(".txt")
    return [list(map(float, l.split())) for l in t.read_text().split("\n") if l.strip()] if t.exists() else []


def make_tile(job):
    img_path, out_stem, seed, negative = job
    rng = np.random.default_rng(seed)
    im = cv2.imread(str(img_path))
    H, W = im.shape[:2]
    labels = read_labels(Path(img_path))
    boxes = np.array([[(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H] for _, x, y, w, h in labels])
    cls = [int(l[0]) for l in labels]
    if negative:                                   # crop away from the labeled sign
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
    th, tw = tile.shape[:2]
    lines = []
    for c, (a0, b0, a1, b1) in zip(cls, boxes):
        a0, a1 = np.clip([a0 - x0, a1 - x0], 0, tw)
        b0, b1 = np.clip([b0 - y0, b1 - y0], 0, th)
        if a1 - a0 > 2 and b1 - b0 > 2:
            lines.append(f"{c} {(a0 + a1) / 2 / tw:.6f} {(b0 + b1) / 2 / th:.6f} {(a1 - a0) / tw:.6f} {(b1 - b0) / th:.6f}")
    cv2.imwrite(f"{out_stem}.jpg", tile)
    Path(f"{out_stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def main():
    names = [l for l in (BAL / "classes.txt").read_text().split("\n") if l]
    rnd = random.Random(0)
    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ("train", "val"):
        (OUT / "images" / s).mkdir(parents=True)
        (OUT / "labels" / s).mkdir(parents=True)

    train, val, val_full, syn_val_classes = [], [], [], []
    for cdir in sorted(p for p in BAL.iterdir() if p.is_dir() and not p.name.startswith("_")):
        real = sorted(cdir.glob("real_*.jpg"))
        syn = sorted(cdir.glob("syn_*.jpg"))
        if len(real) >= 3:
            n_val = max(1, int(round(VAL_SHARE * len(real))))
            start = rnd.randint(0, len(real) - n_val)
            v = real[start:start + n_val]
            vt = [p for p in real if p not in set(v)] + syn
            val += v; val_full += v
        elif not real and syn:
            n_val = max(2, int(round(SYN_VAL_SHARE * len(syn))))
            v = syn[-n_val:]; vt = syn[:-n_val]
            val += v; syn_val_classes.append(cdir.name)
        else:
            vt = real + syn
        train += vt
    if SUBSET < 1:
        train = rnd.sample(train, int(SUBSET * len(train)))

    jobs = []
    for split, files in (("train", train), ("val", val)):
        for i, p in enumerate(files):
            stem = f"{p.parent.name.replace(' ', '_')}__{p.stem}"
            jobs.append((p, str(OUT / "images" / split / stem), i, False))
            if split == "train" and rnd.random() < TILE_NEG_SHARE:
                jobs.append((p, str(OUT / "images" / split / (stem + "_neg")), i + 10 ** 7, True))
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as pool:
        pool.map(make_tile, jobs, chunksize=64)
    for split in ("train", "val"):                 # move .txt files into labels/
        for t in (OUT / "images" / split).glob("*.txt"):
            t.rename(OUT / "labels" / split / t.name)

    (OUT / "val_full.txt").write_text("\n".join(str(p) for p in val_full) + "\n")
    (OUT / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(OUT), "train": "images/train", "val": "images/val", "names": dict(enumerate(names))},
        sort_keys=False, allow_unicode=True))
    (OUT / "data_val_full.yaml").write_text(yaml.safe_dump(
        {"path": str(OUT), "train": "images/train", "val": str(OUT / "val_full.txt"), "names": dict(enumerate(names))},
        sort_keys=False, allow_unicode=True))
    print(f"train images {len(train)}, val images {len(val)} (real {len(val_full)}), "
          f"classes with synthetic-only val: {len(syn_val_classes)}")


if __name__ == "__main__":
    main()
