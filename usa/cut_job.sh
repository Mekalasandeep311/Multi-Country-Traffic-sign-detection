#!/bin/bash
#SBATCH --job-name=us_cut
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cut_%j.log
# Best 5-minute window of the v4 clip2 detection video (most distinct classes): 01:25-06:25.
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/Sandeep_USA
F=/data/common_datasets/ai24mtech02003/envs/sg_signs/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
$F -hide_banner -y -ss 85 -i video_test_v4/clip2_detected.mp4 -t 300 -c:v libx264 -preset medium -crf 21 -pix_fmt yuv420p -movflags +faststart -an video_test_v4/clip2_best5min_01m25-06m25_v4.mp4
ls -la video_test_v4/clip2_best5min_01m25-06m25_v4.mp4
