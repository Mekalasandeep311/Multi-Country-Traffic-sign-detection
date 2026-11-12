"""Map Mapillary Traffic Sign Dataset (MTSD v2, fully annotated) labels onto our 209 handbook classes.

MTSD labels carry no country, only design groups (<name>--gN); which group is the US design must be seen.
  python mtsd_map.py sheet   contact sheet per candidate MTSD label: our handbook template + up to 9 MTSD crops
                             (from the extracted val images) -> mtsd/map_sheets/<our class>.jpg, mtsd/candidates.csv
After the sheets are checked, the accepted "label -> class id" pairs go into MAPPING_OK (written by hand).
"""
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).parent
MT = ROOT / "mtsd"
ANN = MT / "ann" / "mtsd_v2_fully_annotated"
IMG = MT / "images"                      # extracted val (and later train) jpgs

# MTSD label base name (without --gN) -> our class id (0-based). Candidates only; the group is checked on the sheet.
CANDIDATES = {
    "warning--turn-left": 92, "warning--turn-right": 93,
    "warning--curve-left": 94, "warning--curve-right": 94,
    "warning--double-turn-first-right": 96,
    "warning--double-curve-first-left": 97, "warning--double-curve-first-right": 97,
    "warning--winding-road-first-left": 98, "warning--winding-road-first-right": 98,
    "warning--hairpin-curve-left": 99, "warning--hairpin-curve-right": 99,
    "warning--loop-270-degree": 102,
    "warning--horizontal-alignment-left": 104, "warning--horizontal-alignment-right": 104,
    "warning--crossroads": 105,
    "warning--junction-with-a-side-road-perpendicular-left": 106,
    "warning--junction-with-a-side-road-perpendicular-right": 106,
    "warning--junction-with-a-side-road-acute-left": 107, "warning--junction-with-a-side-road-acute-right": 107,
    "warning--t-roads": 108, "warning--y-roads": 109, "warning--offset-roads": 110,
    "warning--roundabout": 111,
    "warning--stop-ahead": 113,
    "warning--traffic-signals": 115,
    "warning--traffic-merges-left": 118, "warning--traffic-merges-right": 118,
    "warning--road-narrows": 119, "warning--added-lane-right": 120,
    "warning--road-narrows-left": 123, "warning--road-narrows-right": 123,
    "warning--pass-left-or-right": 125,
    "warning--steep-ascent": 126,
    "warning--slippery-road-surface": 129,
    "warning--falling-rocks-or-debris-right": 131,
    "warning--railroad-crossing": 134, "warning--railroad-crossing-with-barriers": 134,
    "warning--railroad-crossing-without-barriers": 134, "warning--railroad-intersection": 135,
    "warning--trams-crossing": 136,
    "warning--emergency-vehicles": 139,
    "warning--children": 140,
    "warning--bicycles-crossing": 141,
    "warning--equestrians-crossing": 149,
    "warning--pedestrians-crossing": 150,
    "warning--domestic-animals": 151,
    "warning--wild-animals": 153,
    "warning--height-restriction": 155,
    "warning--playground": 164,
    "warning--roadworks": 165,
    "warning--school-zone": 84,
    "complementary--maximum-speed-limit-15": 157, "complementary--maximum-speed-limit-20": 157,
    "complementary--maximum-speed-limit-25": 157, "complementary--maximum-speed-limit-30": 157,
    "complementary--maximum-speed-limit-35": 157, "complementary--maximum-speed-limit-40": 157,
    "complementary--maximum-speed-limit-45": 157, "complementary--maximum-speed-limit-50": 157,
    "complementary--maximum-speed-limit-55": 157,
    "complementary--one-direction-left": 133, "complementary--one-direction-right": 133,
    "regulatory--stop": 0, "regulatory--yield": 2,
    "regulatory--maximum-speed-limit-15": 9, "regulatory--maximum-speed-limit-20": 9,
    "regulatory--maximum-speed-limit-25": 9, "regulatory--maximum-speed-limit-30": 9,
    "regulatory--maximum-speed-limit-35": 9, "regulatory--maximum-speed-limit-40": 9,
    "regulatory--maximum-speed-limit-45": 9, "regulatory--maximum-speed-limit-50": 9,
    "regulatory--maximum-speed-limit-55": 9, "regulatory--maximum-speed-limit-65": 9,
    "regulatory--maximum-speed-limit-70": 9,
    "information--minimum-speed-40": 15,
    "regulatory--no-right-turn": 16, "regulatory--no-left-turn": 17, "regulatory--no-u-turn": 18,
    "regulatory--left-turn-yield-on-green": 19,
    "regulatory--bicycles-only": 22,
    "regulatory--no-overtaking": 25,
    "regulatory--keep-right": 27, "regulatory--keep-left": 28,
    "regulatory--no-entry": 30, "regulatory--wrong-way": 30,
    "regulatory--no-heavy-goods-vehicles": 31, "regulatory--no-bicycles": 32,
    "regulatory--one-way-straight": 35, "regulatory--one-way-left": 36, "regulatory--one-way-right": 36,
    "information--parking": 39,
    "regulatory--no-parking": 45,
    "regulatory--no-stopping": 59,
    "complementary--tow-away-zone": 50,
    "regulatory--no-pedestrians": 74,
    "regulatory--turning-vehicles-yield-to-pedestrians": 76,
    "regulatory--stop-here-on-red-or-flashing-light": 78,
    "regulatory--no-turn-on-red": 79,
    "regulatory--road-closed": 81,
    "regulatory--weight-limit": 166,
    "information--interstate-route": 182,
}


def base(label):
    return re.sub(r"--g\d+$", "", label)


def load_ann(split_keys=None):
    """(key, json) for every fully annotated image (optionally only the given keys)."""
    for f in glob.glob(str(ANN / "annotations" / "*.json")):
        k = Path(f).stem
        if split_keys is None or k in split_keys:
            yield k, json.load(open(f))


def find_image(key):
    p = IMG / f"{key}.jpg"
    return p if p.exists() else None


def sheet():
    names = yaml.safe_load((ROOT / "class_map.yaml").read_text())["names"]
    tdirs = {int(p.name.split("_")[0]) - 1: p for p in (ROOT / "templates").iterdir() if p.is_dir()}   # folders are 1-based rows
    crops = defaultdict(list)          # label -> [crop]
    counts = defaultdict(int)
    for k, a in load_ann():
        for o in a["objects"]:
            pr = o["properties"]
            if pr.get("dummy") or pr.get("ambiguous") or base(o["label"]) not in CANDIDATES:
                continue
            counts[o["label"]] += 1
            if len(crops[o["label"]]) >= 9 or pr.get("occluded"):
                continue
            p = find_image(k)
            if p is None:
                continue
            b = o["bbox"]
            if min(b["xmax"] - b["xmin"], b["ymax"] - b["ymin"]) < 24:
                continue
            im = cv2.imread(str(p))
            x0, y0, x1, y1 = (int(b[c]) for c in ("xmin", "ymin", "xmax", "ymax"))
            m = int(0.15 * max(x1 - x0, y1 - y0))
            c = im[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m]
            if c.size:
                crops[o["label"]].append(cv2.resize(c, (96, 96)))
    out = MT / "map_sheets"
    out.mkdir(exist_ok=True)
    by_class = defaultdict(list)
    for lab in counts:
        by_class[CANDIDATES[base(lab)]].append(lab)
    rows = []
    for cid, labs in sorted(by_class.items()):
        t = cv2.imread(str(next(tdirs[cid].glob("*.png"))), cv2.IMREAD_UNCHANGED)
        if t.shape[2] == 4:
            a = t[:, :, 3:] / 255.0
            t = (t[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
        t = cv2.resize(t, (96, 96))
        lines = []
        for lab in sorted(labs):
            tiles = crops[lab] + [np.zeros((96, 96, 3), np.uint8)] * (9 - len(crops[lab]))
            head = np.full((96, 260, 3), 255, np.uint8)
            cv2.putText(head, lab.replace("regulatory--", "reg--").replace("complementary--", "comp--")[:40],
                        (3, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1)
            cv2.putText(head, f"{counts[lab]} boxes (train+val)", (3, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 200), 1)
            lines.append(np.hstack([t, head] + tiles))
            rows.append([cid, names[cid], lab, counts[lab]])
        title = np.full((24, lines[0].shape[1], 3), 230, np.uint8)
        cv2.putText(title, f"{cid} {names[cid]}", (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imwrite(str(out / f"{cid:03d}.jpg"), np.vstack([title] + lines), [cv2.IMWRITE_JPEG_QUALITY, 85])
    with open(MT / "candidates.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class_id", "class", "mtsd_label", "boxes"])
        w.writerows(rows)
    print(f"{len(rows)} MTSD labels -> {len(by_class)} of our classes; sheets in {out}")


# Checked on the contact sheets (2026-10-31): only MTSD groups that show the US design of our class.
MAPPING_OK = {
    **{f"regulatory--maximum-speed-limit-{v}--{g}": 9 for v, g in
       [(25, "g2"), (30, "g3"), (35, "g2"), (40, "g3"), (45, "g3"), (55, "g2"), (65, "g2")]},
    "regulatory--bicycles-only--g2": 22, "regulatory--bicycles-only--g3": 22,
    "regulatory--keep-right--g4": 27, "regulatory--keep-left--g2": 28,
    "regulatory--one-way-left--g2": 35, "regulatory--one-way-right--g2": 35,
    "regulatory--one-way-left--g3": 36, "regulatory--one-way-right--g3": 36,
    "regulatory--no-stopping--g4": 45,                       # US "NO PARKING" text plates
    "regulatory--turning-vehicles-yield-to-pedestrians--g1": 76,
    "regulatory--stop-here-on-red-or-flashing-light--g1": 78, "regulatory--stop-here-on-red-or-flashing-light--g2": 78,
    "regulatory--no-turn-on-red--g1": 79, "regulatory--no-turn-on-red--g2": 79, "regulatory--no-turn-on-red--g3": 79,
    # regulatory--road-closed--g2 (-> 81) dropped: MTSD group mixes ROAD CLOSED with other text plates
    "warning--school-zone--g2": 84,
    "warning--turn-left--g1": 92, "warning--turn-right--g1": 93,
    "warning--curve-left--g2": 94, "warning--curve-right--g2": 94,
    "warning--double-turn-first-right--g1": 96,
    "warning--double-curve-first-left--g2": 97, "warning--double-curve-first-right--g2": 97,
    "warning--winding-road-first-left--g1": 98, "warning--winding-road-first-right--g1": 98,
    "warning--hairpin-curve-left--g1": 99, "warning--hairpin-curve-right--g1": 99, "warning--hairpin-curve-right--g4": 99,
    "warning--loop-270-degree--g1": 102,
    "warning--horizontal-alignment-left--g1": 104, "warning--horizontal-alignment-right--g1": 104,
    "warning--crossroads--g3": 105,
    "warning--junction-with-a-side-road-perpendicular-left--g3": 106,
    "warning--junction-with-a-side-road-perpendicular-right--g3": 106,
    "warning--junction-with-a-side-road-acute-left--g1": 107, "warning--junction-with-a-side-road-acute-right--g1": 107,
    "warning--t-roads--g2": 108, "warning--y-roads--g1": 109, "warning--offset-roads--g3": 110,
    "warning--roundabout--g25": 111, "warning--stop-ahead--g9": 113, "warning--traffic-signals--g3": 115,
    "warning--traffic-merges-left--g1": 118, "warning--traffic-merges-right--g1": 118,
    "warning--road-narrows-left--g2": 119, "warning--road-narrows-right--g2": 119,
    "warning--added-lane-right--g1": 120, "warning--road-narrows--g2": 123,
    "warning--pass-left-or-right--g2": 125, "warning--steep-ascent--g7": 126,
    "warning--slippery-road-surface--g2": 129, "warning--falling-rocks-or-debris-right--g2": 131,
    "complementary--one-direction-left--g1": 133, "complementary--one-direction-right--g1": 133,
    "warning--railroad-crossing--g1": 134,
    "warning--railroad-intersection--g3": 135, "warning--railroad-intersection--g4": 135,
    "warning--emergency-vehicles--g1": 139, "warning--bicycles-crossing--g2": 141,
    "warning--pedestrians-crossing--g4": 150, "warning--domestic-animals--g3": 151, "warning--wild-animals--g4": 153,
    "warning--height-restriction--g2": 155,
    **{f"complementary--maximum-speed-limit-{v}--g1": 157 for v in (15, 20, 25, 30, 35, 40, 45, 50, 55)},
    "warning--playground--g3": 164,
    "information--interstate-route--g1": 182,
}


def count():
    """Boxes and images per our class from the accepted mapping (train+val, no test labels)."""
    names = yaml.safe_load((ROOT / "class_map.yaml").read_text())["names"]
    splits = {}
    for s in ("train", "val"):
        for k in open(ANN / "splits" / f"{s}.txt").read().split():
            splits[k] = s
    boxes, imgs = defaultdict(int), defaultdict(set)
    for k, a in load_ann(set(splits)):
        for o in a["objects"]:
            pr = o["properties"]
            cid = MAPPING_OK.get(o["label"])
            if cid is None or pr.get("dummy") or pr.get("ambiguous"):
                continue
            boxes[cid] += 1
            imgs[cid].add(k)
    real = {}
    with open(ROOT / "autobox" / "report.csv") as fh:
        for r in csv.DictReader(fh):
            real[int(r["class_id"])] = int(r["frames"])
    with open(MT / "coverage.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class_id", "class", "drive_frames", "mtsd_images", "mtsd_boxes"])
        for cid in range(len(names)):
            w.writerow([cid, names[cid], real.get(cid, 0), len(imgs[cid]), boxes[cid]])
    mt = {c for c in range(len(names)) if imgs.get(c)}
    both = sum(1 for c in range(len(names)) if real.get(c, 0) or c in mt)
    new = sum(1 for c in mt if not real.get(c, 0))
    print(f"MTSD covers {len(mt)} classes ({new} with no drive frames); "
          f"classes with any real data: {both} of {len(names)}; MTSD images used: {len(set().union(*imgs.values()))}")


if __name__ == "__main__":
    {"sheet": sheet, "count": count}[sys.argv[1] if len(sys.argv) > 1 else "sheet"]()
