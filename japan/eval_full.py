"""Honest full-frame evaluation of a model + top confusion pairs (input for hard-mining).

The first full-frame eval (full_train_job.sh) pointed YOLO at the standard_japan images, so each
val frame was scored against its own folder's .txt only: one sign per frame. Here every val frame
gets the label set used in training:
  folder - union of all folder labels of that frame (kept + extra copies in standard_japan)
  all    - folder + the round-2b "other" labels (autobox/round2_other.csv, model conf >= 0.70)
"all" is the fairer precision estimate but its extra labels come from the model itself, so it
leans optimistic; "folder" leans pessimistic. The truth is in between.

Outputs runs/<name>_eval/full_{folder,all}/ (plots), per_class_full_{folder,all}.csv,
confusions_{640,full_all}.csv (gt class, predicted class, count, share of that gt class)
usage: [SPLIT=jp_train] python eval_full.py [weights] [name]
"""
import csv
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).parent
BAL = ROOT / "standard_japan"
WEIGHTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "jp_signs_v2" / "weights" / "best.pt"
NAME = sys.argv[2] if len(sys.argv) > 2 else "jp_signs_v2"
EV = ROOT / "runs" / f"{NAME}_eval"
OUT = ROOT / "jp_eval_full"
SPLIT = ROOT / os.environ.get("SPLIT", "jp_train")


def read_lines(t):
    return {l.strip() for l in t.read_text().split("\n") if l.strip()} if t.exists() else set()


def build():
    union = defaultdict(set)
    for p in list(BAL.glob("*/real_*.jpg")) + list(BAL.glob("*/extra/real_*.jpg")):
        union[p.name[len("real_"):]] |= {" ".join(f"{float(v):g}" if i == 0 else f"{float(v):.6f}"
                                                  for i, v in enumerate(l.split()))
                                         for l in read_lines(p.with_suffix(".txt"))}
    other = defaultdict(set)
    oc = ROOT / "autobox" / "round2_other.csv"
    if oc.exists():
        for r in csv.DictReader(open(oc)):
            other[r["file"]].add(f"{r['class_id']} {float(r['xc']):.6f} {float(r['yc']):.6f} "
                                 f"{float(r['w']):.6f} {float(r['h']):.6f}")
    val = [Path(l) for l in (SPLIT / "val_full.txt").read_text().split("\n") if l]
    names = [l for l in (BAL / "classes.txt").read_text().split("\n") if l]
    shutil.rmtree(OUT, ignore_errors=True)
    n_other = 0
    for kind in ("folder", "all"):
        (OUT / kind / "images").mkdir(parents=True)
        (OUT / kind / "labels").mkdir(parents=True)
        seen = set()
        for p in val:
            key = p.name[len("real_"):]
            if key in seen:                        # same frame in val of two classes: score it once
                continue
            seen.add(key)
            stem = f"{p.parent.name.replace(' ', '_')}__{p.stem}"
            (OUT / kind / "images" / f"{stem}.jpg").symlink_to(p.resolve())
            lines = union[key] | (other[key] if kind == "all" else set())
            n_other += len(other[key]) if kind == "all" else 0
            (OUT / kind / "labels" / f"{stem}.txt").write_text("".join(f"{l}\n" for l in sorted(lines)))
        (OUT / f"{kind}.yaml").write_text(yaml.safe_dump(
            {"path": str(OUT / kind), "train": "images", "val": "images", "names": dict(enumerate(names))},
            sort_keys=False, allow_unicode=True))
    print(f"full-frame val: {len(seen)} unique frames, +{n_other} other labels in 'all'")


def confusions(cm, names, path, min_count=2):
    """cm[pred, gt]; last row/col = background. Writes the off-diagonal class pairs."""
    m = cm.matrix
    nc = len(names)
    gt_tot = m[:, :nc].sum(0)
    rows = []
    for g in range(nc):
        for p_ in range(nc + 1):
            if p_ != g and m[p_, g] >= min_count:
                rows.append([names[g], names[p_] if p_ < nc else "(missed)", int(m[p_, g]),
                             round(float(m[p_, g] / max(gt_tot[g], 1)), 3)])
    for p_ in range(nc):                           # detections with no ground truth
        if m[p_, nc] >= min_count:
            rows.append(["(background)", names[p_], int(m[p_, nc]), ""])
    rows.sort(key=lambda r: -r[2])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gt_class", "predicted_as", "count", "share_of_gt"])
        w.writerows(rows)
    return rows


def main():
    build()
    model = YOLO(str(WEIGHTS))
    names = [model.names[i] for i in range(len(model.names))]
    EV.mkdir(parents=True, exist_ok=True)
    for tag, data, sz in (("640", SPLIT / "data.yaml", 640),
                          ("full_folder", OUT / "folder.yaml", 1920),
                          ("full_all", OUT / "all.yaml", 1920)):
        r = model.val(data=str(data), imgsz=sz, batch=8, split="val", plots=True, verbose=False,
                      project=str(EV), name=tag, exist_ok=True)
        print(f"[{tag}] P {r.box.mp:.3f} R {r.box.mr:.3f} mAP50 {r.box.map50:.3f} mAP50-95 {r.box.map:.3f}", flush=True)
        with open(EV / f"per_class_{tag}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["class_id", "class", "P", "R", "AP50", "AP50-95"])
            for i, c in enumerate(r.box.ap_class_index):
                p, rr, ap50, ap = r.box.class_result(i)
                w.writerow([int(c), names[int(c)], f"{p:.3f}", f"{rr:.3f}", f"{ap50:.3f}", f"{ap:.3f}"])
        if tag != "full_folder":
            rows = confusions(r.confusion_matrix, names, EV / f"confusions_{tag}.csv")
            print(f"  top class confusions [{tag}]:")
            for g, p_, n, s in [x for x in rows if x[0] != "(background)" and x[1] != "(missed)"][:15]:
                print(f"    {g} -> {p_}: {n} ({s})")


if __name__ == "__main__":
    main()
