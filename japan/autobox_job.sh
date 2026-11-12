#!/bin/bash
#SBATCH --job-name=jp_autobox
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/autobox_%j.log

# Japan stage 2: auto-box the folder's sign in every Merged image (.txt beside each image),
# cross-check with the synthetic YOLO, restore safe demotions, then build standard_japan/ (400/class).
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/sandeep_japan
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
export HF_HOME=$PWD/models

nvidia-smi --query-gpu=name,memory.total --format=csv
python -u autobox_real.py
python -u crosscheck_yolo.py
python -u restore_crosscheck.py
rm -rf autobox_backup_round1 && cp -r autobox autobox_backup_round1     # round-1 state for relabel_round2.py
python -u build_balanced_dataset.py 400
