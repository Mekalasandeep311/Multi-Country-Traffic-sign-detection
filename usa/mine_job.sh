#!/bin/bash
#SBATCH --job-name=us_mine
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mine_%j.log

# clip2 sign tracks for hand labelling (mine_clip2.py props + tracks). Labels are decided by review, not here.
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/Sandeep_USA
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
export HF_HOME=$PWD/models

python -u mine_clip2.py props
python -u mine_clip2.py tracks
