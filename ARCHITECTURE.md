# Architecture and technical stack

## Pipeline (same for all 4 countries)

```
 Country rule book (PDF / XLSX / DOCX)
        │  extract_templates.py / make_templates.py
        ▼
 Class list (one class per handbook sign)  +  clean sign template per class
        │
        │      Real drive footage (dash-cam, test-bed videos, YouTube drives, nuScenes, MTSD)
        │              │  frame sampling
        ▼              ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ Auto-boxing  (autobox_real.py / autolabel.py)                │
 │  1. OWLv2 open-vocabulary detector proposes "traffic sign"   │
 │     boxes                                                    │
 │  2. each crop is classified against the templates            │
 │     (DINOv2 embeddings + a synthetic-trained YOLO)           │
 │  3. accept >= 0.30, uncertain 0.10-0.30 -> review sheets     │
 └──────────────────────────────────────────────────────────────┘
        │  audit_labels.py, shape_check.py, crosscheck_yolo.py, contact sheets checked by eye
        ▼
 Clean real labels  ──►  build_balanced_dataset.py
                          • cap per class (300-400 images)
                          • classes with few/no real photos: template pasted onto real
                            background frames (sign swap) with blur, light, perspective
                          • sign-free negative frames
                          • 640 px native-resolution tiles from 1920 px frames
        │
        ▼
 YOLO11s training (SLURM, A40 / H200)
   lr check (0.005 / 0.01 / 0.02) -> full run with early stopping,
   validation on REAL frames only (no synthetic data in val / test)
        │
        ▼
 Evaluation on held-out full-size frames (eval_full.py, rescore_job.sh)
   COCO-style matching, bootstrap 95 % CI, per-class CSVs
        │  calibrate_thresholds.py (per-class confidence on real val frames)
        ▼
 Video inference (detect.py): YOLO11s + ByteTrack + per-class thresholds,
   track kept once confirmed so labels do not flicker
        │  mine_video.py / mine_india.py: wrong or missed tracks reviewed by eye
        └──────────────► fed back as new labelled frames (next model version)
```

## Model

- **Detector:** Ultralytics YOLO11s (9.4 M parameters), single-stage, anchor-free, P3-P5 heads.
- **Why 640 tiles at native resolution:** signs are often < 30 px in a 1920 px frame.
  Training on full-resolution tiles keeps them large enough, and inference runs on the whole frame at 1920 (India: 1280).
- **Training recipe (all countries):** SGD, lr0 0.005, cosine LR, 3 warm-up epochs, mosaic 1.0
  (off for the last 10 epochs), mixup 0.1, HSV saturation 0.6 / value 0.5, scale 0.5, **horizontal flip off**
  (mirrored signs change meaning: keep left vs keep right), AMP.
- **Hidden "Other sign" class (India, USA):** trained on real and synthetic signs that are not in the rule book,
  so they are not forced into the nearest handbook class. It is never shown in the output.

## Country notes

| Country | Data sources | Special steps |
|---|---|---|
| India | Real Indian road images, TiHAN test-bed videos (1920x1200) | v2 adds frames mined from test-bed videos (new classes: width limit, charging station, ...); v3 fine-tunes for "No parking"; benchmark of 8 detectors |
| Japan | Real drive frames merged from several folders | Duplicate-name check, label audit, per-class thresholds |
| Singapore | YouTube drive videos, nuScenes Singapore (Boston scenes removed) | Hard-negative and video fine-tuning, 512x288 real-time model (ONNX) |
| USA | Drive videos, Mapillary MTSD (48 mapped classes) | Synthetic out-of-list plates for "Other sign", video track review (v4) |

## Technical stack

| Part | Tools |
|---|---|
| Language | Python 3.11, Bash |
| Deep learning | PyTorch 2.5.1 (CUDA 12.1), torchvision |
| Detector | Ultralytics 8.4.174 (YOLO11s; benchmarks YOLOv8s, YOLOv10s, YOLO26-L, RT-DETR-L) |
| Auto-labelling | Hugging Face Transformers: OWLv2 (`google/owlv2-base-patch16-ensemble`), DINOv2 (`facebook/dinov2-small/base`) |
| Images / video | OpenCV, Pillow, ffmpeg (H.264 encode) |
| Tracking | ByteTrack (Ultralytics tracker) |
| Data / config | NumPy, PyYAML, JSON, CSV |
| Documents | PyMuPDF (`fitz`) for cutting sign templates out of the rule-book PDFs |
| Compute | SLURM cluster, NVIDIA A40 and H200 GPUs, conda environment |
| Deployment | PyTorch `.pt`, ONNX (Singapore 512x288); TensorRT FP16 export planned |
