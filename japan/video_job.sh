#!/bin/bash
#SBATCH --job-name=jp_video
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/video_%j.log

# Tokyo video test on the held-out part (minutes 15-30): native 1280x720 and upscaled x1.5
# (signs then appear at the same pixel size as in the 1920x1080 training frames).
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/sandeep_japan
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
export HF_HOME=$PWD/models

python - <<'PY'
import csv
rows = list(csv.DictReader(open("standard_japan/report.csv")))
open("synthetic_only_classes.txt", "w").write("".join(r["class"] + "\n" for r in rows if int(r["real_used"]) == 0))
print("synthetic-only classes:", sum(int(r["real_used"]) == 0 for r in rows))
PY
mkdir -p video_test
for S in 1 1.5; do
  START_S=900 END_S=1800 SCALE=$S python -u detect.py tokya_trimmed.mp4 runs/jp_signs_v2/weights/best.pt video_test
done
