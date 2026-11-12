#!/bin/bash
#SBATCH --job-name=jp_hmprep
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/hm_prep_%j.log

# Hard-mining prep: keep the delivered dataset as standard_japan_v1, rebuild standard_japan with the
# round-2 labels, and make 200 targeted synthetic images for each often-confused class.
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/sandeep_japan
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
export HF_HOME=$PWD/models

[ -e standard_japan_v1 ] || mv standard_japan standard_japan_v1
python -u build_balanced_dataset.py 400
python -u verify_standard.py
# Hours Regulated, Start/Within/End control zone plates, End of control, No Stopping or Parking,
# Elderly-stopping vs Stopping Allowed, Fire Hydrant vs Time-restricted Parking, No Overtaking,
# Road Closed to Pedestrians, Pedestrians and Bicycles Only, Road Closed to Light Road Vehicles,
# Prohibition of Vehicle Entry, Designates Lane for Destination, Destination & Distance
SYN_ONLY=$(echo 33 101 119 21 22 51 105 103 28 106 49 86 62 83 74 11 15 | tr ' ' '\n' | sed 's/$/:200/' | paste -sd,)
BALANCED_OUT=$PWD/jp_hardmine/targeted_syn SYN_ONLY=$SYN_ONLY python -u build_balanced_dataset.py
