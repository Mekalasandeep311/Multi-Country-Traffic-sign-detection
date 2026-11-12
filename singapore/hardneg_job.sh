#!/bin/bash
#SBATCH --job-name=sg_hardneg
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/hardneg_%j.log

# Hard-negative fine-tune of runs/sg_signs_v2/weights/best.pt, then old-vs-new comparison.
set -uo pipefail
cd /data/common_datasets/ai24mtech02003/sandeep_singapore
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs

python -u build_hardneg_finetune.py
python -u - <<'EOF'
from ultralytics import YOLO
YOLO("runs/sg_signs_v2/weights/best.pt").train(
    data="hardneg/ft_data.yaml", name="sg_signs_v2_hn", project="runs", exist_ok=True,
    epochs=30, imgsz=640, batch=64, device=0, workers=8,
    optimizer="SGD", lr0=0.001, lrf=0.1, momentum=0.937, weight_decay=5e-4, warmup_epochs=1, cos_lr=True,
    patience=10, close_mosaic=5, mosaic=1.0, mixup=0.1, hsv_v=0.5, scale=0.5, translate=0.1,
    fliplr=0.0, flipud=0.0, degrees=0.0)
EOF
python -u - <<'EOF'
import numpy as np, cv2
from pathlib import Path
from ultralytics import YOLO
hold = [l for l in Path("hardneg/clean_holdout.txt").read_text().split("\n") if l]
for tag, w in (("OLD best.pt", "runs/sg_signs_v2/weights/best.pt"), ("NEW hard-neg", "runs/sg_signs_v2_hn/weights/best.pt")):
    m = YOLO(w)
    v = m.val(data="sg_signs_train/data.yaml", imgsz=640, batch=32, plots=False, verbose=False, name="cmp", project="runs/hn_eval", exist_ok=True)
    f = m.val(data="sg_signs_train/data_val_full.yaml", imgsz=1920, batch=8, plots=False, verbose=False, name="cmp", project="runs/hn_eval", exist_ok=True)
    fp = {t: 0 for t in (0.25, 0.5)}
    for p in hold:
        im = cv2.imread(p)
        c = m.predict(im, imgsz=int(np.ceil(max(im.shape[:2]) / 32) * 32), conf=0.25, verbose=False)[0].boxes.conf.tolist()
        for t in fp: fp[t] += sum(x >= t for x in c)
    print(f"{tag}: val tiles mAP50 {v.box.map50:.3f} mAP50-95 {v.box.map:.3f} R {v.box.mr:.3f} | "
          f"full frames mAP50 {f.box.map50:.3f} R {f.box.mr:.3f} | clean holdout ({len(hold)} frames) "
          f"FP@0.25 {fp[0.25]} FP@0.5 {fp[0.5]}")
EOF
