"""Train YOLO11 on jp_train (real + synthetic, real-only val).

usage: python train_v2.py <data.yaml> <run_name> <lr0> <epochs> [model] [batch]

Against overfitting: early stopping on val (patience), best.pt kept, mosaic/mixup/HSV/scale
augmentation, balanced classes. Learning rate: SGD lr0 (chosen by the short LR check), 3 warmup
epochs, cosine decay to lr0*lrf, weight decay 5e-4.
"""
import os
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).parent
data, name, lr0, epochs = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
model = sys.argv[5] if len(sys.argv) > 5 else "yolo11s.pt"
batch = int(sys.argv[6]) if len(sys.argv) > 6 else 64

YOLO(str(ROOT / model)).train(
    data=data, name=name, project=str(ROOT / "runs"), exist_ok=True,
    epochs=epochs, imgsz=640, batch=batch,
    device=0 if torch.cuda.is_available() else "cpu",
    workers=min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", 4))),
    optimizer="SGD", lr0=lr0, lrf=0.01, momentum=0.937, weight_decay=5e-4,
    cos_lr=True,
    patience=int(os.environ.get("PATIENCE", 25)), close_mosaic=10,
    warmup_epochs=float(os.environ.get("WARMUP", 3)),
    mosaic=1.0, mixup=0.1, hsv_h=0.015, hsv_s=0.6, hsv_v=0.5, scale=0.5, translate=0.1,
    fliplr=0.0, flipud=0.0, degrees=0.0,   # mirrored / rotated signs change meaning
    plots=True,
)
