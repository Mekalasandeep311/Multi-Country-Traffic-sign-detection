#!/bin/bash
#SBATCH --job-name=in_video
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/video_%j.log
# India: test-bed videos with india_v1 (hidden Other sign not drawn), contact sheets, H.264 copies for download.
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/Sandeep_indiatraffic
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
F=$(python -c "import imageio_ffmpeg as i; print(i.get_ffmpeg_exe())")
mkdir -p video_test_v1
for v in cam_20261106_100001 cam_20261106_100849; do
  python -u detect.py infer_video/$v.mp4 checkpoints/india_v1_best.pt video_test_v1
  python -u video_sheet.py video_test_v1/${v}_detections.csv infer_video/$v.mp4 video_test_v1/${v}_sheet
  $F -hide_banner -loglevel error -y -i video_test_v1/${v}_detected.mp4 -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p -movflags +faststart -an video_test_v1/${v}_v1_h264.mp4
done
ls -la video_test_v1
