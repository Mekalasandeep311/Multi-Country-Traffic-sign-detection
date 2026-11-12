#!/bin/bash
#SBATCH --job-name=jp_lr
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/lr_check_%j.log

# Short learning-rate check: 3 x 15 epochs on a 20% subset, lr0 in {0.005, 0.01, 0.02}.
set -uo pipefail
cd /data/common_datasets/ai24mtech02003/sandeep_japan
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
export HF_HOME=$PWD/models

[ -f jp_train_20pct/data.yaml ] || python -u make_train_split.py 0.2
for lr in 0.005 0.01 0.02; do
    python -u train_v2.py jp_train_20pct/data.yaml "lr_check_${lr}" "$lr" 15
done
python - <<'EOF'
import csv
for lr in ("0.005", "0.01", "0.02"):
    rows = list(csv.DictReader(open(f"runs/lr_check_{lr}/results.csv")))
    k = {c.strip(): c for c in rows[0]}
    best = max(float(r[k["metrics/mAP50-95(B)"]]) for r in rows)
    last = rows[-1]
    print(f"lr0={lr}: best val mAP50-95 {best:.3f}  last train box/cls {float(last[k['train/box_loss']]):.3f}/"
          f"{float(last[k['train/cls_loss']]):.3f}  val box/cls {float(last[k['val/box_loss']]):.3f}/{float(last[k['val/cls_loss']]):.3f}")
EOF
