#!/bin/bash
#SBATCH --job-name=jp_hm
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/hm_train_%j.log

# Hard-mining fine-tune of jp_signs_v2 and a fair comparison (same real val frames as v2).
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/sandeep_japan
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
export HF_HOME=$PWD/models

python -u mine_video.py tiles video_test/tokya_trimmed_0-15min_detections.csv tokya_trimmed.mp4 jp_hardmine/video
SPLIT_OUT=jp_train_hm FIXED_VAL=jp_train/val_full.txt OTHER_LABELS=1 \
  EXTRA_SYN=jp_hardmine/targeted_syn EXTRA_TILES=jp_hardmine/video/tiles python -u make_train_split.py 1.0
PATIENCE=15 WARMUP=1 python -u train_v2.py jp_train_hm/data.yaml jp_signs_v2_hm 0.001 40 runs/jp_signs_v2/weights/best.pt
# both models on the same (new) val: tiles 640 + full real frames 1920
SPLIT=jp_train_hm python -u eval_full.py runs/jp_signs_v2/weights/best.pt jp_signs_v2_on_hmval
SPLIT=jp_train_hm python -u eval_full.py runs/jp_signs_v2_hm/weights/best.pt jp_signs_v2_hm
# held-out video minutes 15-30 with the new model
mkdir -p video_test_hm
START_S=900 END_S=1800 python -u detect.py tokya_trimmed.mp4 runs/jp_signs_v2_hm/weights/best.pt video_test_hm
