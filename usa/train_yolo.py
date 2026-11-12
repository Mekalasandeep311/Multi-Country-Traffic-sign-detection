"""Train YOLO11 on the synthetic USA sign dataset (dataset/data.yaml).

usage: python train_yolo.py [epochs] [model] [batch]   e.g. python train_yolo.py 80 yolo11s.pt 64
"""
import os
import sys

import torch
from ultralytics import YOLO

from autolabel import ROOT

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
weights = sys.argv[2] if len(sys.argv) > 2 else "yolo11n.pt"
batch = int(sys.argv[3]) if len(sys.argv) > 3 else 16

model = YOLO(str(ROOT / weights))
model.train(
    data=str(ROOT / "dataset" / "data.yaml"),
    imgsz=640,              # tiles are cut at native resolution; detect.py runs on the full frame
    epochs=epochs,
    batch=batch,
    device=0 if torch.cuda.is_available() else "cpu",
    workers=min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", 2))),
    fliplr=0.0,             # never mirror: a flipped "turn left" is a "turn right"
    degrees=0.0,
    scale=0.3,
    hsv_v=0.5,
    mosaic=1.0,
    close_mosaic=5,
    patience=15,
    project=str(ROOT / "runs"),
    name="us_synth",
    exist_ok=True,
)
