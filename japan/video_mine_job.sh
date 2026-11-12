#!/bin/bash
#SBATCH --job-name=jp_vmine
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/video_mine_%j.log

# Tokyo video minutes 0-15 (the mining half) at native resolution with the v2 model, then group the
# detections into sign passes and draw review sheets.
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/sandeep_japan
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
export HF_HOME=$PWD/models

START_S=0 END_S=900 python -u detect.py tokya_trimmed.mp4 runs/jp_signs_v2/weights/best.pt video_test
python -u mine_video.py events video_test/tokya_trimmed_0-15min_detections.csv tokya_trimmed.mp4 jp_hardmine/video
