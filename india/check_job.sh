#!/bin/bash
#SBATCH --job-name=in_check
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/check_%j.log
# India label check: render rule-book signs (+ mirrored variants) -> train YOLO11s-cls -> classify every box.
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/Sandeep_indiatraffic
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
python -u render_cls.py 300
cp -n /data/common_datasets/ai24mtech02003/Sandeep_USA/yolo11s.pt . 2>/dev/null || true
yolo classify train data=cls_data model=yolo11s-cls.yaml pretrained=False imgsz=128 epochs=40 batch=256 \
     fliplr=0 flipud=0 project=runs name=label_check exist_ok=True workers=12 plots=False
python -u classify_crops.py runs/label_check/weights/best.pt
