"""Turn reviewed video detections (video_mine/<stem>/detections.csv) into fine-tune data.

Every detection is classified by rules written from a visual review of the per-class sheets:
  pos  real sign of that class (or RELABEL target)  -> YOLO label
  fp   not a sign / not a class we have            -> left unlabeled (hard negative)
  unc  unclear                                       -> the whole frame is skipped
Frames whose detections are all pos/fp are used. The last HOLDOUT share of the video (by time)
is kept as an evaluation set; the rest becomes 640x640 native-scale training tiles (one centred
on each positive, one on each FP region).

Output: video_ft/{images,labels}/{train,val}, video_ft/data_video_val.yaml, video_ft/summary.txt
usage: python build_video_finetune.py [stem]
"""
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).parent
STEM = sys.argv[1] if len(sys.argv) > 1 else "video1_last30min"
MINE = ROOT / "video_mine" / STEM
OUT = ROOT / "video_ft"
TILE = 640
HOLDOUT = 0.2

# class -> min conf to count as positive; below it: fp if listed in FP_BELOW else unc
POS_MIN = {
    "Keep left": 0.5, "No entry for all vehicles": 0.6, "Pedestrian crossing prohibition": 0.5,
    "Advance Directional Sign": 0.5, "Confirmation Directional Sign": 0.5, "Split traffic": 0.5,
    "Curve Alignment Markers": 0.4, "No right turn": 0.4, "U-turn sign with vehicle weight restriction": 0.55,
    "Electronic Road Pricing (ERP) Zone ahead": 0.5, "Road Closure (with time)": 0.6,
    "Height limit (4.5m)": 0.64, "No stopping": 0.8, "Give Way to pedestrians": 0.5,
    "Parking Reinforcement Camera": 0.8, "Works Ahead": 0.25, "Light signals ahead": 0.6,
    "Speed limit 60 km": 0.3, "One-way street": 0.93, "Right Turn (for specific vehicles)": 1.01,
}
FP_BELOW = {"Height limit (4.5m)": 0.64, "No stopping": 0.4, "Keep left": 0.3}
RELABEL = {"Speed limit 60 km": "Speed limit 50 km"}   # all '60' detections were 50 signs
FP_CLASSES = {
    "Ambulance", "Bicycle stand", "Caution Road Works Ahead", "Detour Ahead", "Must Exit", "No Parking",
    "No sounding of horn", "Rain shelter for motorcyclists", "Roadside Variable Message Sign (RVMS)",
    "Traffic Information Display (TID)", "Traffic Sign Display (TSD)", "End of school zone", "Gantry Sign",
    "Lane Reduction (lane 4)", "Parking Paid", "Pedestrian overpass", "Waiting Area for Motorycyclists",
    "Wheel Clamp Zone", "Red Cross (lane closed)", "Speed Check Area", "Traffic signals operating hours",
    "Turn left ahead", "White walking sticks", "Exit Closed Ahead",
    "Flashing Amber Cross (caution, possible obstruction)", "Green Arrow (lane open)",
    "Turn on lights and FM Radio", "Stop for children",
}
UNREVIEWED_POS_MIN = 0.85   # classes not reviewed visually: only very confident boxes are trusted


def classify(d, W):
    cls, conf = d["class"], float(d["conf"])
    if cls in FP_CLASSES:
        return "fp"
    if cls == "Give Way to pedestrians" and float(d["x0"]) < 0.01 * W:
        return "fp"                                   # object cut off at the left frame edge
    if cls in POS_MIN:
        if conf >= POS_MIN[cls]:
            return "pos"
        return "fp" if conf < FP_BELOW.get(cls, -1) else "unc"
    return "pos" if conf >= UNREVIEWED_POS_MIN else "unc"


def main():
    names = yaml.safe_load((ROOT / "sg_signs_train" / "data.yaml").read_text())["names"]
    cid = {v: int(k) for k, v in names.items()}
    with open(MINE / "detections.csv") as fh:
        dets = list(csv.DictReader(fh))
    by_frame = {}
    for d in dets:
        by_frame.setdefault(int(d["frame"]), []).append(d)
    frames = sorted(by_frame)
    cut = frames[int((1 - HOLDOUT) * len(frames))]

    shutil.rmtree(OUT, ignore_errors=True)
    for s in ("train", "val"):
        (OUT / "images" / s).mkdir(parents=True)
        (OUT / "labels" / s).mkdir(parents=True)

    stats = Counter()
    pos_cls = Counter()
    rng = np.random.default_rng(0)
    for fr in frames:
        im = cv2.imread(str(MINE / "frames" / f"{fr:06d}.jpg"))
        H, W = im.shape[:2]
        tags = [(classify(d, W), d) for d in by_frame[fr]]
        if any(t == "unc" for t, _ in tags):
            stats["frames skipped (uncertain)"] += 1
            continue
        pos = [(cid[RELABEL.get(d["class"], d["class"])], [float(d[k]) for k in ("x0", "y0", "x1", "y1")])
               for t, d in tags if t == "pos"]
        fps = [[float(d[k]) for k in ("x0", "y0", "x1", "y1")] for t, d in tags if t == "fp"]
        for c, _ in pos:
            pos_cls[names[c]] += 1
        stats["pos boxes"] += len(pos); stats["fp boxes"] += len(fps)

        if fr >= cut:                                 # held-out: full frame
            cv2.imwrite(str(OUT / "images" / "val" / f"{fr:06d}.jpg"), im)
            (OUT / "labels" / "val" / f"{fr:06d}.txt").write_text("".join(
                f"{c} {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}\n"
                for c, b in pos))
            stats["val frames"] += 1
            continue

        centres = [("p", b) for _, b in pos] + [("n", b) for b in fps]
        for k, (kind, b) in enumerate(centres):
            cx, cy = (b[0] + b[2]) / 2 + rng.uniform(-150, 150), (b[1] + b[3]) / 2 + rng.uniform(-150, 150)
            x0 = int(np.clip(cx - TILE / 2, 0, max(0, W - TILE)))
            y0 = int(np.clip(cy - TILE / 2, 0, max(0, H - TILE)))
            tile = im[y0:y0 + TILE, x0:x0 + TILE]
            th, tw = tile.shape[:2]
            lines = []
            for c, (a0, b0, a1, b1) in pos:
                ca0, ca1 = np.clip([a0 - x0, a1 - x0], 0, tw)
                cb0, cb1 = np.clip([b0 - y0, b1 - y0], 0, th)
                if (ca1 - ca0) * (cb1 - cb0) >= 0.5 * (a1 - a0) * (b1 - b0):   # mostly inside
                    lines.append(f"{c} {(ca0 + ca1) / 2 / tw:.6f} {(cb0 + cb1) / 2 / th:.6f} "
                                 f"{(ca1 - ca0) / tw:.6f} {(cb1 - cb0) / th:.6f}")
                elif ca1 > ca0 and cb1 > cb0:          # partly cut sign: grey it out instead of mislabeling
                    tile = tile.copy()
                    tile[int(cb0):int(cb1), int(ca0):int(ca1)] = 114
            name = f"v{fr:06d}_{k}_{kind}"
            cv2.imwrite(str(OUT / "images" / "train" / f"{name}.jpg"), tile)
            (OUT / "labels" / "train" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            stats[f"train tiles ({kind})"] += 1

    (OUT / "data_video_val.yaml").write_text(yaml.safe_dump(
        {"path": str(OUT), "train": "images/train", "val": "images/val", "names": names},
        sort_keys=False, allow_unicode=True))
    text = "\n".join(f"{k}: {v}" for k, v in sorted(stats.items()))
    text += "\npositives per class: " + ", ".join(f"{k} {v}" for k, v in pos_cls.most_common())
    (OUT / "summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
