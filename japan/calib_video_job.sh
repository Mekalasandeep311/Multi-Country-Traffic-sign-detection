#!/bin/bash
#SBATCH --job-name=jp_calib
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/calib_video_%j.log

# Inference-only improvement of v2: per-class thresholds from real val at video size, then the
# held-out video minutes 15-30 again with the new thresholds + sticky tracks.
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/sandeep_japan
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
export HF_HOME=$PWD/models

python -u calibrate_thresholds.py runs/jp_signs_v2/weights/best.pt 1280
mkdir -p video_test_v2cal
START_S=900 END_S=1800 python -u detect.py tokya_trimmed.mp4 runs/jp_signs_v2/weights/best.pt video_test_v2cal
python -u video_sheet.py video_test_v2cal/tokya_trimmed_15-30min_detections.csv tokya_trimmed.mp4 video_test_v2cal/sheet
