#!/bin/bash
#SBATCH --job-name=jp_full
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/full_train_%j.log

# Full training: 120-class balanced data, real-only val, lr0 chosen by the LR check
# (best val mAP50-95 over runs/lr_check_*), up to 150 epochs with early stopping.
set -uo pipefail
cd /data/common_datasets/ai24mtech02003/sandeep_japan
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
export HF_HOME=$PWD/models

python -u make_train_split.py 1.0
LR=$(python - <<'EOF'
import csv, glob
best = None
for d in glob.glob("runs/lr_check_*"):
    rows = list(csv.DictReader(open(f"{d}/results.csv")))
    k = {c.strip(): c for c in rows[0]}
    m = max(float(r[k["metrics/mAP50-95(B)"]]) for r in rows)
    if best is None or m > best[0]:
        best = (m, d.split("_")[-1])
print(best[1])
EOF
)
echo "chosen lr0 = $LR"
python -u train_v2.py jp_train/data.yaml jp_signs_v2 "$LR" 150
python -u - <<'EOF'
# Evaluation for hard-mining later: per-class P/R/AP + confusion matrix (which signs get mixed up).
import csv
from pathlib import Path
from ultralytics import YOLO
ROOT = Path.cwd()
m = YOLO("runs/jp_signs_v2/weights/best.pt")
for name, data, sz in (("val tiles (real + synthetic-only classes)", "jp_train/data.yaml", 640),
                       ("full real frames", "jp_train/data_val_full.yaml", 1920)):
    r = m.val(data=data, imgsz=sz, batch=8, split="val", plots=True, verbose=False, name=f"val_{sz}",
              project=str(ROOT / "runs" / "jp_signs_v2_eval"), exist_ok=True)
    print(f"{name}: P {r.box.mp:.3f} R {r.box.mr:.3f} mAP50 {r.box.map50:.3f} mAP50-95 {r.box.map:.3f}")
    with open(ROOT / "runs" / "jp_signs_v2_eval" / f"per_class_{sz}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class_id", "class", "P", "R", "AP50", "AP50-95"])
        for i, c in enumerate(r.box.ap_class_index):
            p, rr, ap50, ap = r.box.class_result(i)
            w.writerow([int(c), m.names[int(c)], f"{p:.3f}", f"{rr:.3f}", f"{ap50:.3f}", f"{ap:.3f}"])
EOF
