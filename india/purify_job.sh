#!/bin/bash
#SBATCH --job-name=in_pur
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/purify_%j.log
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/Sandeep_indiatraffic
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
python -u purify_tracks.py 0.55
for v in cam_20261106_100001 cam_20261106_100849; do VID=$v python mine_india.py write; done
python lab_check.py
