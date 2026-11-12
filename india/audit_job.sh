#!/bin/bash
#SBATCH --job-name=in_audit
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/audit_%j.log
# India: DINOv2 audit of every labelled box vs the 98 rule-book pictures (read-only on the source data).
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/Sandeep_indiatraffic
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
export HF_HOME=/data/common_datasets/ai24mtech02003/Sandeep_USA/models
python -u audit_crops.py
