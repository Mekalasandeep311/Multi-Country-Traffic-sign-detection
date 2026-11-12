#!/bin/bash
#SBATCH --job-name=in_inv
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/inventory_%j.log
# India: rule-book templates + inventory of the two read-only source datasets (no source file is changed).
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/Sandeep_indiatraffic
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
python -u make_templates.py
python -u inventory.py 16
python -u - <<'PY'
import json, collections
rows = [json.loads(l) for l in open("inventory.jsonl")]
k = collections.Counter(r["kind"] for r in rows); bad = sum(1 for r in rows if r.get("bad"))
md5 = collections.Counter(r.get("md5") for r in rows if not r.get("bad"))
print("kinds", dict(k), "unreadable", bad, "exact duplicates", sum(c - 1 for c in md5.values() if c > 1))
PY
