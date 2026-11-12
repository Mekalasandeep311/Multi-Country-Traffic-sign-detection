#!/bin/bash
#SBATCH --job-name=us_autobox
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/autobox_%j.log

# USA stage 2: auto-box every sign in the real frames (OWLv2 + synthetic YOLO must agree), one A40.
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/Sandeep_USA
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
export HF_HOME=$PWD/models
python -u autobox_us.py all
