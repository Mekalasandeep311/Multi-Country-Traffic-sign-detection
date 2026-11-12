#!/bin/bash
#SBATCH --job-name=us_video
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/video_%j.log

# USA video test: runs/us_signs_v2 on infer_video/clip2.mp4 (1280x720, native scale = training frames).
# NOTE: clip2 is a cut of the training drive video3_0 (New York); its labelled sign frames were trained on,
# so this is a demo + a test for the synthetic-only classes, not a fully held-out test.
set -euo pipefail
cd /data/common_datasets/ai24mtech02003/Sandeep_USA
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
export HF_HOME=$PWD/models

python - <<'PY'
import csv
rows = list(csv.DictReader(open("standard_usa/report.csv")))
open("synthetic_only_classes.txt", "w").write("".join(r["class"] + "\n" for r in rows if int(r["real_used"]) == 0))
print("synthetic-only classes:", sum(int(r["real_used"]) == 0 for r in rows))
PY
mkdir -p video_test
python -u detect.py infer_video/clip2.mp4 runs/us_signs_v2/weights/best.pt video_test
python -u video_sheet.py video_test/clip2_detections.csv infer_video/clip2.mp4 video_test/clip2_sheet
python - <<'PY'
import csv
from collections import defaultdict
synth = set(open("synthetic_only_classes.txt").read().splitlines())
rows = list(csv.DictReader(open("video_test/clip2_detections.csv")))
by = defaultdict(list)
for r in rows:
    by[r["class_name"]].append(r)
print(f"{len(rows)} boxes, {len(by)} classes")
for k in sorted(by, key=lambda k: -len(by[k])):
    rs = by[k]
    print(f"  {k}{' [synthetic-only]' if k in synth else ''}: {len(rs)} boxes, "
          f"t {float(rs[0]['time_s']):.0f}-{float(rs[-1]['time_s']):.0f}s, max conf {max(float(r['conf']) for r in rs):.2f}")
PY
