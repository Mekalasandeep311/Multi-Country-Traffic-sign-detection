#!/bin/bash
#SBATCH --job-name=jp_signs
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.log

# Japan stage 1: backgrounds (+OWLv2 cache) -> synthetic dataset -> YOLO11s class scorer, one A40.
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/sandeep_japan
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
export HF_HOME=$PWD/models

nvidia-smi --query-gpu=name,memory.total --format=csv
N_TRAIN=${N_TRAIN:-20000}
N_VAL=${N_VAL:-1000}
EPOCHS=${EPOCHS:-80}

python -u make_class_map.py
python -u make_backgrounds.py 400
python -u make_yolo_dataset.py "$N_TRAIN" "$N_VAL"
python -u train_yolo.py "$EPOCHS" yolo11s.pt 64
