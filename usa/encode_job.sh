#!/bin/bash
#SBATCH --job-name=us_enc
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/encode_%j.log
# Re-encode the v4 clip2 detection video to H.264 (small, plays in any player) for download.
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/Sandeep_USA
F=/data/common_datasets/ai24mtech02003/envs/sg_signs/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
$F -hide_banner -y -i video_test_v4/clip2_detected.mp4 -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p -movflags +faststart -an video_test_v4/clip2_detected_v4_h264.mp4
ls -la video_test_v4/clip2_detected_v4_h264.mp4
