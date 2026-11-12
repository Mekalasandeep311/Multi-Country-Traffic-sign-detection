#!/bin/bash
#SBATCH --job-name=jp_build
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/build_%j.log

# Shape check on the current labels, snapshot round-1 state, rebuild standard_japan (400/class).
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/sandeep_japan
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
export HF_HOME=$PWD/models

python -u shape_check.py
rm -rf autobox_backup_round1 && cp -r autobox autobox_backup_round1
python -u build_balanced_dataset.py 400
