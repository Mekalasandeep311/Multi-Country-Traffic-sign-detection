"""Build a synthetic YOLO detection dataset from the handbook templates.

Backgrounds are the real Singapore dashcam frames in this folder. Any sign OWLv2
already found there (output/cache/*.json from autolabel.py) is inpainted away first,
so un-labeled real signs are not taught as "background". Handbook signs are then
pasted in at realistic sizes, angles and lighting, so every box label is exact.

Tiles are 640x640 cut at native resolution, so small signs keep their real pixel size;
detect.py then runs on the full-resolution frame.

usage: python make_yolo_dataset.py [n_train] [n_val]
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from autolabel import ROOT, load_template_rgba, photometric, warp_sign

OUT = ROOT / "dataset"
TILE = 640


def load_backgrounds():
    bgs = []
    for p in sorted(ROOT.glob("*.jpg")):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        cache = ROOT / "output" / "cache" / f"{p.stem}.json"
        if cache.exists():
            mask = np.zeros(img.shape[:2], np.uint8)
            for d in json.loads(cache.read_text()):
                x0, y0, x1, y1 = d["box"]
                m = 0.2 * max(x1 - x0, y1 - y0)
                cv2.rectangle(mask, (int(x0 - m), int(y0 - m)), (int(x1 + m), int(y1 + m)), 255, -1)
            img = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)
        else:
            print(f"warning: no detection cache for {p.name}; its real signs stay unlabeled")
        bgs.append(img)
    return bgs


def load_templates():
    temps = []
    for p in sorted((ROOT / "templates").glob("*/*.png")):
        cid = int(p.parent.name.split("_")[0])
        temps.append((cid, load_template_rgba(p)))
    return temps


def random_tile(bgs, rng):
    bg = bgs[rng.integers(len(bgs))]
    s = rng.uniform(max(0.6, TILE / min(bg.shape[:2]) + 0.01), 1.4)
    bg = cv2.resize(bg, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    H, W = bg.shape[:2]
    x, y = rng.integers(0, W - TILE + 1), rng.integers(0, H - TILE + 1)
    return bg[y:y + TILE, x:x + TILE].astype(np.float32).copy()


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def make_tile(bgs, temps, order, rng):
    tile = random_tile(bgs, rng)
    labels, boxes = [], []
    n = 0 if rng.random() < 0.1 else int(rng.integers(2, 9))   # 10% pure negatives
    for _ in range(n):
        cid, rgba = temps[next(order)]
        # small signs are far more common in dashcam frames
        target_h = int(np.exp(rng.uniform(np.log(14), np.log(150))))
        aspect = rgba.shape[1] / rgba.shape[0]
        target_h = min(target_h, int(320 / aspect))
        sign = warp_sign(rgba, max(10, target_h), rng)
        a = sign[..., 3]
        ys, xs = np.nonzero(a > 127)
        if len(xs) == 0:
            continue
        sign = sign[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        sh, sw = sign.shape[:2]
        if sw >= TILE or sh >= TILE:
            continue
        for _try in range(10):
            x, y = int(rng.integers(0, TILE - sw)), int(rng.integers(0, TILE - sh))
            box = (x, y, x + sw, y + sh)
            if all(iou(box, b) < 0.05 for b in boxes):
                break
        else:
            continue
        rgb = sign[..., :3].astype(np.float32) * rng.uniform(0.5, 1.15) + rng.uniform(-20, 20)
        al = sign[..., 3:4].astype(np.float32) / 255
        tile[y:y + sh, x:x + sw] = tile[y:y + sh, x:x + sw] * (1 - al) + np.clip(rgb, 0, 255) * al
        boxes.append(box)
        labels.append(f"{cid - 1} {(x + sw / 2) / TILE:.6f} {(y + sh / 2) / TILE:.6f} "
                      f"{sw / TILE:.6f} {sh / TILE:.6f}")
    return photometric(tile.astype(np.uint8), rng), labels


def class_cycle(temps, rng):
    """Yield template indices so every class appears equally often."""
    by_class = {}
    for i, (cid, _) in enumerate(temps):
        by_class.setdefault(cid, []).append(i)
    cids = list(by_class)
    while True:
        rng.shuffle(cids)
        for c in cids:
            yield by_class[c][rng.integers(len(by_class[c]))]


def main(n_train=4000, n_val=400):
    rng = np.random.default_rng(0)
    shutil.rmtree(OUT, ignore_errors=True)
    bgs, temps = load_backgrounds(), load_templates()
    order = class_cycle(temps, rng)
    for split, n in (("train", n_train), ("val", n_val)):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
        for i in range(n):
            img, labels = make_tile(bgs, temps, order, rng)
            cv2.imwrite(str(OUT / "images" / split / f"{i:05d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            (OUT / "labels" / split / f"{i:05d}.txt").write_text("\n".join(labels))
        print(f"{split}: {n} tiles")

    classes = {int(k): v for k, v in json.loads((ROOT / "classes.json").read_text()).items()}
    classes.setdefault(85, "Right Turn (for specific vehicles)")
    names = {i - 1: classes[i] for i in range(1, max(classes) + 1)}
    (OUT / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(OUT), "train": "images/train", "val": "images/val", "names": names},
        sort_keys=False, allow_unicode=True))
    print("wrote", OUT / "data.yaml")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]))
