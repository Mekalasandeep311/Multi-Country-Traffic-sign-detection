#!/bin/bash
#SBATCH --job-name=in_cls
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/classify_%j.log
# India label check: classify every labelled box with the label-check classifier (read-only on source data).
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/Sandeep_indiatraffic
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
python -u classify_crops.py runs/classify/runs/label_check/weights/best.pt
