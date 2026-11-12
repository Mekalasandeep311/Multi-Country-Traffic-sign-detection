#!/bin/bash
#SBATCH --job-name=jp_fix
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/fix_%j.log

# Re-select labels from the cached candidates with the Japan fixes, then rebuild standard_japan.
#   Rest Area folder -> class "Rest Area - Parking - Fuel Stand"; plate classes boxed on the plate;
#   same object labeled in two folders -> weaker label demoted.
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/sandeep_japan
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
export HF_HOME=$PWD/models

python -u make_class_map.py
rm -f "autobox/cache/Rest Area_yolo.npz"          # was scored for the P+cup class; rescore for Fuel Stand
python -u autobox_real.py select
python -u fix_labels_jp.py
python -u crosscheck_yolo.py
python -u restore_crosscheck.py
python -u shape_check.py
rm -rf autobox_backup_round1 && cp -r autobox autobox_backup_round1
python -u build_balanced_dataset.py 400
