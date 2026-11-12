#!/bin/bash
#SBATCH --job-name=in_build
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/build_%j.log
# India: standard_india (99 rule-book classes x 400, real first, sign swap for the rest) + real-only val/test split.
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/Sandeep_indiatraffic
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
python -u build_standard.py 32
