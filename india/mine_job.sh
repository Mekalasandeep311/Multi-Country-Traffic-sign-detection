#!/bin/bash
#SBATCH --job-name=in_mine
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mine_%j.log
# India: sign tracks from the two test-bed videos (OWLv2 + india_v1), review sheets for eye review.
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/Sandeep_indiatraffic
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
export HF_HOME=/data/common_datasets/ai24mtech02003/Sandeep_USA/models
for v in cam_20261106_100001 cam_20261106_100849; do
  VID=$v python -u mine_india.py props
  VID=$v python -u mine_india.py tracks
done
