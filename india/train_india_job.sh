#!/bin/bash
#SBATCH --job-name=in_train
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_%j.log
# India v1: YOLO11s (COCO pre-trained) on standard_india (99 rule-book classes + hidden Other), real-only val,
# 1280 px, lr0 0.005 (best in the SG / JP / US LR checks), no flips, early stopping; then evaluation.
set -uo pipefail
cd /data/common_datasets/ai24mtech02003/Sandeep_indiatraffic
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
python -u make_split_india.py
cp -n /data/common_datasets/ai24mtech02003/Sandeep_USA/yolo11s.pt . || true
BATCH=${BATCH:-16}
yolo detect train data=india_train/data.yaml model=yolo11s.pt imgsz=1280 epochs=120 patience=20 batch=$BATCH \
     lr0=0.005 lrf=0.01 cos_lr=True warmup_epochs=3 optimizer=SGD fliplr=0 flipud=0 degrees=0 mosaic=1.0 \
     close_mosaic=10 mixup=0.1 workers=16 cache=False project=runs name=india_v1 exist_ok=True plots=True
python -u - <<'EOF'
import csv, json
from ultralytics import YOLO
m = YOLO("runs/detect/runs/india_v1/weights/best.pt") if __import__("os").path.exists("runs/detect/runs/india_v1/weights/best.pt") else YOLO("runs/india_v1/weights/best.pt")
for split in ("val", "test"):
    r = m.val(data="india_train/data.yaml", imgsz=1280, batch=8, split=split, plots=True, verbose=False,
              project="runs", name=f"india_v1_eval_{split}", exist_ok=True)
    print(f"{split}: P {r.box.mp:.3f} R {r.box.mr:.3f} mAP50 {r.box.map50:.3f} mAP50-95 {r.box.map:.3f}")
    with open(f"runs/india_v1_per_class_{split}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["class_id", "class", "P", "R", "AP50", "AP50-95"])
        for i, c in enumerate(r.box.ap_class_index):
            p, rr, a50, a = r.box.class_result(i); w.writerow([int(c), m.names[int(c)], f"{p:.3f}", f"{rr:.3f}", f"{a50:.3f}", f"{a:.3f}"])
EOF
