#!/bin/bash
#SBATCH --job-name=us_full
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/full_train_%j.log

# USA full training: 209-class standard_usa, real-only val (US drives), MTSD-val held out as test,
# lr0 = best val mAP50-95 of runs/us_lr_check_*, up to 150 epochs with early stopping (patience 25).
# Launch only after lr_check_job.sh has finished.
set -uo pipefail
cd /data/common_datasets/ai24mtech02003/Sandeep_USA
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
export HF_HOME=$PWD/models

[ -f us_train/data.yaml ] || python -u make_train_split.py 1.0
LR=$(python - <<'EOF'
import csv, glob
best = None
for d in glob.glob("runs/us_lr_check_*"):
    rows = list(csv.DictReader(open(f"{d}/results.csv")))
    k = {c.strip(): c for c in rows[0]}
    m = max(float(r[k["metrics/mAP50-95(B)"]]) for r in rows)
    if best is None or m > best[0]:
        best = (m, d.split("_")[-1])
print(best[1])
EOF
)
echo "chosen lr0 = $LR"
python -u train_v2.py us_train/data.yaml us_signs_v2 "$LR" 150
python -u - <<'EOF'
# Over/underfitting check + evaluation: per-class P/R/AP and the most frequent class confusions.
import csv
from pathlib import Path
import numpy as np
from ultralytics import YOLO
ROOT = Path.cwd()
rows = list(csv.DictReader(open("runs/us_signs_v2/results.csv")))
k = {c.strip(): c for c in rows[0]}
b = max(range(len(rows)), key=lambda i: float(rows[i][k["metrics/mAP50-95(B)"]]))
for tag, i in (("best", b), ("last", len(rows) - 1)):
    r = rows[i]
    print(f"{tag} epoch {i + 1}: val mAP50-95 {float(r[k['metrics/mAP50-95(B)']]):.3f}  train box/cls "
          f"{float(r[k['train/box_loss']]):.3f}/{float(r[k['train/cls_loss']]):.3f}  val box/cls "
          f"{float(r[k['val/box_loss']]):.3f}/{float(r[k['val/cls_loss']]):.3f}")
m = YOLO("runs/us_signs_v2/weights/best.pt")
ev = ROOT / "runs" / "us_signs_v2_eval"
for name, data, split, sz, bs in (("val tiles (real + synthetic-only classes)", "us_train/data.yaml", "val", 640, 16),
                                  ("full real US drive frames", "us_train/data_val_full.yaml", "val", 1280, 8),
                                  ("MTSD test (held out, never trained)", "us_train/data.yaml", "test", 1920, 4)):
    tag = f"{split}_{sz}"
    r = m.val(data=data, imgsz=sz, batch=bs, split=split, plots=True, verbose=False, name=tag,
              project=str(ev), exist_ok=True)
    print(f"{name}: P {r.box.mp:.3f} R {r.box.mr:.3f} mAP50 {r.box.map50:.3f} mAP50-95 {r.box.map:.3f}")
    with open(ev / f"per_class_{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class_id", "class", "P", "R", "AP50", "AP50-95"])
        for i, c in enumerate(r.box.ap_class_index):
            p, rr, ap50, ap = r.box.class_result(i)
            w.writerow([int(c), m.names[int(c)], f"{p:.3f}", f"{rr:.3f}", f"{ap50:.3f}", f"{ap:.3f}"])
    cm = getattr(r, "confusion_matrix", None)
    if cm is not None:                     # matrix[pred, gt]; last row/col = background
        M = cm.matrix[:-1, :-1]
        gt_tot = cm.matrix[:, :-1].sum(0)
        pairs = [(M[p, g], g, p) for g in range(M.shape[1]) for p in range(M.shape[0]) if p != g and M[p, g] > 0]
        with open(ev / f"confusions_{tag}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["gt_class", "predicted_class", "count", "share_of_gt"])
            for n, g, p in sorted(pairs, reverse=True)[:60]:
                w.writerow([m.names[g], m.names[p], int(n), f"{n / max(gt_tot[g], 1):.3f}"])
        wrong = sum(n for n, _, _ in pairs)
        print(f"  class confusions: {int(wrong)} of {int(gt_tot.sum())} gt boxes predicted as another class")
EOF
