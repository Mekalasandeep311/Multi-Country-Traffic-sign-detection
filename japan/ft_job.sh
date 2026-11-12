#!/bin/bash
#SBATCH --job-name=jp_ft
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/ft_%j.log

# Short fine-tune of v2 on audited labels (813 bad real labels dropped, lane-arrow class relabelled by
# hand) + video tiles + other-sign labels, NO targeted synthetic. Same val frames as v2.
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/sandeep_japan
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
export HF_HOME=$PWD/models

# the 8 hand-labelled lane-arrow frames, 5 copies each (different random tiles)
L="jp_hardmine/extra_real/Traffic Distribution by Direction of Proceeding"
rm -rf jp_hardmine/extra_real && mkdir -p "$L"
M="classified Data japan/Merged/Traffic Distribution by Direction of Proceeding"
grep -l . "$M"/*.txt | while read t; do
  b=$(basename "$t" .txt)
  for k in 1 2 3 4 5; do ln "$M/$b.jpg" "$L/lane_${b}_$k.jpg"; cp "$t" "$L/lane_${b}_$k.txt"; done
done
ls "$L" | wc -l

SPLIT_OUT=jp_train_ft FIXED_VAL=jp_train/val_full.txt OTHER_LABELS=1 \
  EXTRA_SYN=jp_hardmine/extra_real EXTRA_TILES=jp_hardmine/video/tiles python -u make_train_split.py 1.0
PATIENCE=10 WARMUP=1 python -u train_v2.py jp_train_ft/data.yaml jp_signs_v2_ft 0.001 30 runs/jp_signs_v2/weights/best.pt
SPLIT=jp_train_ft python -u eval_full.py runs/jp_signs_v2/weights/best.pt jp_signs_v2_on_ftval
SPLIT=jp_train_ft python -u eval_full.py runs/jp_signs_v2_ft/weights/best.pt jp_signs_v2_ft
THRESH_OUT=thresholds_ft.json python -u calibrate_thresholds.py runs/jp_signs_v2_ft/weights/best.pt 1280
mkdir -p video_test_ft
THRESH=thresholds_ft.json START_S=900 END_S=1800 python -u detect.py tokya_trimmed.mp4 runs/jp_signs_v2_ft/weights/best.pt video_test_ft
