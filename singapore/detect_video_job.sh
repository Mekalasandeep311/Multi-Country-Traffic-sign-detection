#!/bin/bash
#SBATCH --job-name=sg_video
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/video_%j.log

# Run best.pt on every video in video/ and save the annotated video, the frames
# that contain signs, and a CSV of all detections next to each video.
set -uo pipefail
cd /data/common_datasets/ai24mtech02003/sandeep_singapore
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs

for v in video/*.mp4 video/*.MP4 video/*.avi video/*.mov video/*.mkv; do
    [ -f "$v" ] || continue
    case "$v" in *_detected.mp4) continue ;; esac
    echo "=== $v"
    python -u detect.py "$v" runs/sg_signs/weights/best.pt video
done
