#!/bin/bash
#SBATCH --job-name=jp_relabel
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/relabel_%j.log

# Round 2 with the v2 model: (a) label unlabeled frames for their own folder class,
# (b) label the other signs in every real frame, (c) honest full-frame eval + confusion pairs.
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/sandeep_japan
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
export HF_HOME=$PWD/models

python -u relabel_round2.py
python -u relabel_other.py
python -u eval_full.py runs/jp_signs_v2/weights/best.pt jp_signs_v2
