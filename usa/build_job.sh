#!/bin/bash
#SBATCH --job-name=us_build
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/build_%j.log

# USA: build standard_usa (400/class): drive frames + MTSD images first, sign-swap synthetic fills the rest.
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/Sandeep_USA
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
export HF_HOME=$PWD/models

python -u build_balanced_dataset.py 400
