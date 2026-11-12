#!/bin/bash
#SBATCH --job-name=sg_signs
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.log

# Synthetic dataset -> YOLO11s training -> sample predictions, all on one A40.
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/sandeep_singapore
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs

nvidia-smi --query-gpu=name,memory.total --format=csv
N_TRAIN=${N_TRAIN:-20000}
N_VAL=${N_VAL:-1000}
EPOCHS=${EPOCHS:-80}

python -u make_yolo_dataset.py "$N_TRAIN" "$N_VAL"
python -u train_yolo.py "$EPOCHS" yolo11s.pt 64
python -u detect.py . runs/sg_signs/weights/best.pt
