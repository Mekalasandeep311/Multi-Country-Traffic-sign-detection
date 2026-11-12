"""Build the hard-negative fine-tune dataset.

Background frames: singapore_Hardmining/ minus frames where best.pt sees anything at conf >= EXCL_CONF
(those contain real signs, e.g. Keep left), minus NEIGHBOURS frames on each side (consecutive video
frames show the same sign). Remaining frames -> two 640x480 native-scale crops each, empty labels.
Frames with near-miss scores (NEAR_LO <= max conf < EXCL_CONF) are repeated NEAR_REPEAT times.
Only frames in split.json "train" are used; "holdout" is kept for the FP evaluation.

Positives: POS_SHARE random subset of sg_signs_train/images/train (keeps all classes).
Backgrounds are repeated until they are BG_SHARE of the training list.

Output: hardneg/ds/{images,labels}/train, hardneg/ft_train.txt, hardneg/ft_data.yaml,
        hardneg/clean_holdout.txt, hardneg/excluded.txt

usage: python build_hardneg_finetune.py
"""
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).parent
SRC = ROOT / "singapore_Hardmining"
HN = ROOT / "hardneg"
TRAIN = ROOT / "sg_signs_train"
WEIGHTS = ROOT / "runs" / "sg_signs_v2" / "weights" / "best.pt"
EXCL_CONF = 0.15
NEAR_LO = 0.05
NEIGHBOURS = 3
NEAR_REPEAT = 3
POS_SHARE = 0.25
BG_SHARE = 0.15


def main():
    frames = sorted(SRC.glob("*.jpg"))
    split = json.loads((HN / "split.json").read_text())
    model = YOLO(str(WEIGHTS))
    maxconf = {}
    for f in frames:
        im = cv2.imread(str(f))
        r = model.predict(im, imgsz=int(np.ceil(max(im.shape[:2]) / 32) * 32), conf=NEAR_LO, verbose=False)[0]
        maxconf[f.name] = max(r.boxes.conf.tolist(), default=0.0)

    names = [f.name for f in frames]
    excluded = set()
    for i, n in enumerate(names):
        if maxconf[n] >= EXCL_CONF:
            excluded.update(names[max(0, i - NEIGHBOURS):i + NEIGHBOURS + 1])
    (HN / "excluded.txt").write_text("\n".join(sorted(excluded)) + "\n")
    clean_train = [n for n in split["train"] if n not in excluded]
    clean_hold = [n for n in split["holdout"] if n not in excluded]
    (HN / "clean_holdout.txt").write_text("\n".join(str(SRC / n) for n in clean_hold) + "\n")

    ds = HN / "ds"
    shutil.rmtree(ds, ignore_errors=True)
    (ds / "images" / "train").mkdir(parents=True)
    (ds / "labels" / "train").mkdir(parents=True)
    rng = np.random.default_rng(0)
    bg = []
    for n in clean_train:
        im = cv2.imread(str(SRC / n))
        H, W = im.shape[:2]
        reps = NEAR_REPEAT if maxconf[n] >= NEAR_LO else 1
        for k in range(2 * reps):
            x0 = int(rng.integers(0, max(1, W - 640 + 1)))
            out = ds / "images" / "train" / f"bg_{Path(n).stem}_{k}.jpg"
            cv2.imwrite(str(out), im[:, x0:x0 + 640])
            (ds / "labels" / "train" / (out.stem + ".txt")).write_text("")
            bg.append(str(out))

    pos = sorted(str(p) for p in (TRAIN / "images" / "train").glob("*.jpg"))
    random.Random(0).shuffle(pos)
    pos = pos[:int(POS_SHARE * len(pos))]
    need = int(BG_SHARE / (1 - BG_SHARE) * len(pos))
    bg_list = [bg[i % len(bg)] for i in range(max(need, len(bg)))]
    # Ultralytics may de-duplicate list entries, so repeated backgrounds get their own hard links
    final_bg = []
    for i, p in enumerate(bg_list):
        if i < len(bg):
            final_bg.append(p); continue
        q = Path(p).with_name(f"{Path(p).stem}_r{i}.jpg")
        if not q.exists():
            q.hardlink_to(p)
            (ds / "labels" / "train" / (q.stem + ".txt")).write_text("")
        final_bg.append(str(q))
    lst = pos + final_bg
    random.Random(1).shuffle(lst)
    (HN / "ft_train.txt").write_text("\n".join(lst) + "\n")

    data = yaml.safe_load((TRAIN / "data.yaml").read_text())
    data["train"] = str(HN / "ft_train.txt")
    data["val"] = str(TRAIN / "images" / "val")
    (HN / "ft_data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    print(f"frames {len(frames)}: excluded (real signs + neighbours) {len(excluded)}, clean train {len(clean_train)}, "
          f"clean holdout {len(clean_hold)}, near-miss frames {sum(1 for n in clean_train if maxconf[n] >= NEAR_LO)}")
    print(f"fine-tune list: {len(pos)} positive tiles + {len(final_bg)} background crops "
          f"({len(final_bg) / len(lst):.1%})")


if __name__ == "__main__":
    main()
