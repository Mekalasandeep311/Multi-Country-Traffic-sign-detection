#!/bin/bash
# One-time: create the conda env and pre-download YOLO weights (compute nodes may lack internet).
ENV=/data/common_datasets/ai24mtech02003/envs/sg_signs
source ~/miniconda3/etc/profile.d/conda.sh
[ -x "$ENV/bin/python" ] || conda create -y -q -p "$ENV" python=3.11 || { echo ENV_FAILED; exit 1; }
conda activate "$ENV"
pip install -q ultralytics transformers pyyaml || { echo ENV_FAILED; exit 1; }
cd /data/common_datasets/ai24mtech02003/sandeep_singapore
python - <<'EOF'
import torch, ultralytics
from ultralytics import YOLO
from ultralytics.utils.checks import check_font
print(torch.__version__, torch.version.cuda, ultralytics.__version__)
YOLO("yolo11s.pt"); YOLO("yolo11n.pt")   # used for training and the AMP check
check_font("Arial.ttf")
EOF
echo ENV_DONE
