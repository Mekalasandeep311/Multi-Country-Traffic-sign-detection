#!/bin/bash
#SBATCH --job-name=sg_full
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/full_train_%j.log

# Full training: 159-class balanced data, real-only val, lr0 chosen by the LR check
# (best val mAP50-95 over runs/lr_check_*), up to 150 epochs with early stopping.
set -uo pipefail
cd /data/common_datasets/ai24mtech02003/sandeep_singapore
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs

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
python -u train_v2.py sg_signs_train/data.yaml sg_signs_v2 "$LR" 150
python -u - <<'EOF'
from ultralytics import YOLO
m = YOLO("runs/sg_signs_v2/weights/best.pt")
for name, data, sz in (("val tiles (real + synthetic-only classes)", "sg_signs_train/data.yaml", 640),
                       ("full real frames", "sg_signs_train/data_val_full.yaml", 1920)):
    r = m.val(data=data, imgsz=sz, batch=8, split="val", plots=False, verbose=False, name=f"val_{sz}", project="runs/sg_signs_v2_eval")
    print(f"{name}: P {r.box.mp:.3f} R {r.box.mr:.3f} mAP50 {r.box.map50:.3f} mAP50-95 {r.box.map:.3f}")
EOF
