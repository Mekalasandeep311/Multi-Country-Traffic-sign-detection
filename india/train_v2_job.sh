#!/bin/bash
#SBATCH --job-name=in_v2
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/train_v2_%j.log
# India v2: fine-tune from the read-only v1 checkpoint with 102 classes (+ Charging station, Width limit) and test-bed frames.
# Then v1 vs v2 on the same real test photos and held-out test-bed blocks, per class.
set -uo pipefail
cd /data/common_datasets/ai24mtech02003/Sandeep_indiatraffic
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
yolo detect train data=india_train_v2/data.yaml model=yolo11s.yaml pretrained=checkpoints/india_v1_best.pt imgsz=1280 \
     epochs=40 patience=12 batch=32 lr0=0.002 lrf=0.05 cos_lr=True warmup_epochs=1 optimizer=SGD fliplr=0 flipud=0 degrees=0 \
     mosaic=1.0 close_mosaic=5 mixup=0.1 workers=16 project=runs name=india_v2 exist_ok=True plots=True
python -u - <<'EOF'
import csv, glob
from ultralytics import YOLO
best = glob.glob("runs/detect/runs/india_v2/weights/best.pt")[0]
for tag, w, sets in (("v1", "checkpoints/india_v1_best.pt", (("real_test", "data_v1.yaml"), ("test_bed", "data_bed_v1.yaml"))),
                     ("v2", best, (("real_test", "data.yaml"), ("test_bed", "data_bed.yaml")))):
    m = YOLO(w)
    for name, y in sets:
        r = m.val(data=f"india_train_v2/{y}", imgsz=1280, batch=8, split="test", plots=False, verbose=False,
                  project="runs", name=f"cmp_{tag}_{name}", exist_ok=True)
        print(f"{tag} {name}: P {r.box.mp:.3f} R {r.box.mr:.3f} mAP50 {r.box.map50:.3f} mAP50-95 {r.box.map:.3f}", flush=True)
        with open(f"runs/cmp_{tag}_{name}.csv", "w", newline="") as fh:
            c = csv.writer(fh); c.writerow(["class_id", "class", "P", "R", "AP50", "AP50-95"])
            for i, k in enumerate(r.box.ap_class_index):
                p, q, a, b = r.box.class_result(i)
                c.writerow([int(k), m.names[int(k)], f"{p:.3f}", f"{q:.3f}", f"{a:.3f}", f"{b:.3f}"])
EOF
