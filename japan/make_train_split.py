"""Train/val split + 640 training tiles from standard_japan.

Val is REAL images only: per class, a contiguous block (by file order, so consecutive near-identical
video frames do not leak between train and val) of ~VAL_SHARE of the real images. Classes with <3
real images get no real val; classes with no real images get a small synthetic val share so they
are still measured (reported separately).

Japan: the same frame can sit in several class folders (one per sign it shows), each copy labeled only
with its folder's sign. For training, every copy gets the UNION of all labels of that frame, and a frame
in val is removed from train entirely.

Tiles: each image -> one 640x640 crop at native resolution that contains its labeled sign(s) at a
random offset (+ TILE_NEG_SHARE extra sign-free crops from real frames as negatives). A sign larger
than the tile is handled by downscaling that image first.

jp_train/images/{train,val}, labels/{train,val}, data.yaml, val_full.txt (full real frames)

Hard-mining options (env, all off by default):
  SPLIT_OUT    output folder instead of jp_train
  FIXED_VAL    a val_full.txt of an earlier split: reuse exactly those real val frames (fair comparison
               with the earlier model) and drop train frames within NEIGHBOR frame numbers of them
  OTHER_LABELS 1 = add the round-2b "other sign" labels (autobox/round2_other.csv) to the frame union
  EXTRA_SYN    folder(s, comma-separated) <class>/*.jpg(+.txt) of extra images (targeted synthetic or
               hand-reviewed real frames), added to train
  EXTRA_TILES  folder of ready 640 tiles (*.jpg + *.txt, e.g. mined from video), added to train

usage: python make_train_split.py [subset_share]   (subset_share < 1 for the LR check)
"""
import csv
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
BAL = ROOT / "standard_japan"
SUBSET = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
OUT = ROOT / os.environ.get("SPLIT_OUT", "jp_train" if SUBSET >= 1 else f"jp_train_{int(SUBSET * 100)}pct")
FIXED_VAL = os.environ.get("FIXED_VAL")
OTHER_LABELS = os.environ.get("OTHER_LABELS") == "1"
EXTRA_SYN = os.environ.get("EXTRA_SYN")
EXTRA_TILES = os.environ.get("EXTRA_TILES")
NEIGHBOR = 30
VAL_SHARE = 0.10
SYN_VAL_SHARE = 0.05
TILE = 640
TILE_NEG_SHARE = 0.10


def read_labels(p):
    t = p.with_suffix(".txt")
    return [list(map(float, l.split())) for l in t.read_text().split("\n") if l.strip()] if t.exists() else []


def frame_key(p):
    """Same raw frame in different class folders: same original name and size."""
    return p.name[len("real_"):], p.stat().st_size


def make_tile(job):
    img_path, out_stem, seed, negative, labels = job
    rng = np.random.default_rng(seed)
    im = cv2.imread(str(img_path))
    H, W = im.shape[:2]
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

    fixed = None
    if FIXED_VAL:
        fixed = {(Path(l).parent.name, Path(l).name) for l in Path(FIXED_VAL).read_text().split(chr(10)) if l}
    train, val, val_full, syn_val_classes = [], [], [], []
    for cdir in sorted(p for p in BAL.iterdir() if p.is_dir() and not p.name.startswith("_")):
        real = [p for p in sorted(cdir.glob("real_*.jpg")) if read_labels(p)]   # empty .txt = dropped by audit
        syn = sorted(cdir.glob("syn_*.jpg"))
        if fixed is not None and real:
            all_real = real + [p for p in sorted((cdir / "extra").glob("real_*.jpg")) if read_labels(p)]
            v = [p for p in all_real if (cdir.name, p.name) in fixed]
            vt = [p for p in real if (cdir.name, p.name) not in fixed] + syn
            val += v; val_full += v
        elif len(real) >= 3:
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
    # union of labels per real frame over all class folders (kept + extra)
    union = {}
    for p in list(BAL.glob("*/real_*.jpg")) + list(BAL.glob("*/extra/real_*.jpg")):
        lines = {" ".join(f"{v:g}" if i == 0 else f"{v:.6f}" for i, v in enumerate(l)) for l in read_labels(p)}
        union.setdefault(frame_key(p), set()).update(lines)
    if OTHER_LABELS:
        by_name = {}
        for k in union:
            by_name.setdefault(k[0], []).append(k)
        n_o = 0
        for r in csv.DictReader(open(ROOT / "autobox" / "round2_other.csv")):
            for k in by_name.get(r["file"], []):
                union[k].add(f"{r['class_id']} {float(r['xc']):.6f} {float(r['yc']):.6f} "
                             f"{float(r['w']):.6f} {float(r['h']):.6f}")
                n_o += 1
        print(f"other-sign labels added to {n_o} frame copies")
    val_keys = {frame_key(p) for p in val_full}
    n0 = len(train)
    train = [p for p in train if not p.name.startswith("real_") or frame_key(p) not in val_keys]
    if fixed is not None:                          # near-identical neighbours of val frames stay out of train
        def num(p):
            d = "".join(ch for ch in p.stem if ch.isdigit())
            return int(d) if d else None
        vnum = np.array(sorted(n for n in (num(p) for p in val_full) if n is not None))
        def near(p):
            n = num(p)
            if n is None or not len(vnum):
                return False
            i = np.searchsorted(vnum, n)
            return any(0 <= j < len(vnum) and abs(vnum[j] - n) <= NEIGHBOR for j in (i - 1, i))
        train = [p for p in train if not p.name.startswith("real_") or not near(p)]
    print(f"removed {n0 - len(train)} train copies of frames that are in (or next to) val")
    if EXTRA_SYN:
        extra_syn = sorted(p for d in EXTRA_SYN.split(",") for p in Path(d).glob("*/*.jpg"))
        train += extra_syn
        print(f"extra images added: {len(extra_syn)}")
    if SUBSET < 1:
        train = rnd.sample(train, int(SUBSET * len(train)))

    def labels_of(p):
        if p.name.startswith("real_"):
            return [list(map(float, l.split())) for l in sorted(union[frame_key(p)])]
        return read_labels(p)

    jobs = []
    for split, files in (("train", train), ("val", val)):
        for i, p in enumerate(files):
            stem = f"{p.parent.name.replace(' ', '_')}__{p.stem}"
            lab = labels_of(p)
            jobs.append((p, str(OUT / "images" / split / stem), i, False, lab))
            if split == "train" and rnd.random() < TILE_NEG_SHARE:
                jobs.append((p, str(OUT / "images" / split / (stem + "_neg")), i + 10 ** 7, True, lab))
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as pool:
        pool.map(make_tile, jobs, chunksize=64)
    if EXTRA_TILES:
        n_t = 0
        for t in sorted(Path(EXTRA_TILES).glob("*.jpg")):
            shutil.copy2(t, OUT / "images" / "train" / f"video__{t.name}")
            shutil.copy2(t.with_suffix(".txt"), OUT / "images" / "train" / f"video__{t.stem}.txt")
            n_t += 1
        print(f"video tiles added: {n_t}")
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
