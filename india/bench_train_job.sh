#!/bin/bash
#SBATCH --job-name=in_bm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/benchtrain_%x_%j.log
# Paper benchmark: popular detector on the same india_train_v2 data and settings, then the common scorer.
# env: TAG (run name), MODEL, PRE (pretrained path or empty), BATCH, EXTRA (extra yolo args)
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
PRETRAIN=""; [ -n "${PRE:-}" ] && PRETRAIN="pretrained=$PRE"
yolo detect train data=india_train_v2/data.yaml model=$MODEL $PRETRAIN imgsz=1280 epochs=100 patience=20 batch=$BATCH \
     fliplr=0 flipud=0 degrees=0 mosaic=1.0 close_mosaic=10 mixup=0.1 workers=16 project=runs name=bench_$TAG exist_ok=True plots=True ${EXTRA:-}
python -u - <<'EOF'
import csv, glob, json, os
from pathlib import Path
import numpy as np, torch
from ultralytics import YOLO, RTDETR
from ultralytics.utils.metrics import ap_per_class
TAG = os.environ["TAG"]
w = glob.glob(f"runs/detect/runs/bench_{TAG}/weights/best.pt") + glob.glob(f"runs/bench_{TAG}/weights/best.pt")
w = w[0]
C = json.load(open("class_map.json"))["classes"]
D = Path("india_train_v2"); OUT = Path("runs/bench"); OUT.mkdir(parents=True, exist_ok=True)
old_train = {Path(n).stem for n in os.listdir("/data/common_datasets/traffic_signs_india_split/images/train")}
def files(i, l, keep=None):
    return [(f, f"{l}/{Path(f).stem}.txt") for f in sorted(glob.glob(f"{i}/*.jpg")) if keep is None or keep(Path(f).stem)]
real = files(D / "images/test", D / "labels/test")
unseen = files(D / "images/test", D / "labels/test", lambda s: s not in old_train)
bed = files(D / "images/test_bed", D / "labels/test_bed")
IOUV = np.linspace(0.5, 0.95, 10)
def iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None); inter = wh[..., 0] * wh[..., 1]
    A = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); B = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (A[:, None] + B[None, :] - inter + 1e-9)
def run(model, sz, fl, tag):
    conf, pcls, tps, tcls = [], [], [], []
    for i in range(0, len(fl), 8):
        part = fl[i:i + 8]
        for (f, lab), r in zip(part, model.predict([p for p, _ in part], imgsz=sz, conf=0.001, iou=0.7, max_det=300, verbose=False, half=True)):
            H, W = r.orig_shape; t = []
            if os.path.exists(lab):
                for l in open(lab):
                    q = l.split()
                    if len(q) >= 5 and int(q[0]) != 99:
                        c, cx, cy, ww, h = int(q[0]), *map(float, q[1:5])
                        t.append([c, (cx - ww / 2) * W, (cy - h / 2) * H, (cx + ww / 2) * W, (cy + h / 2) * H])
            t = np.array(t).reshape(-1, 5)
            b = r.boxes.xyxy.cpu().numpy(); s = r.boxes.conf.cpu().numpy(); k = r.boxes.cls.cpu().numpy().astype(int)
            m = k != 99; b, s, k = b[m], s[m], k[m]
            tp = np.zeros((len(b), 10), bool)
            if len(b) and len(t):
                I = iou(b, t[:, 1:]) * (k[:, None] == t[:, 0][None, :])
                for j, th in enumerate(IOUV):
                    x = np.argwhere(I >= th)
                    if len(x):
                        x = x[np.argsort(-I[x[:, 0], x[:, 1]])]
                        x = x[np.unique(x[:, 0], return_index=True)[1]]
                        x = x[np.argsort(-I[x[:, 0], x[:, 1]])]
                        x = x[np.unique(x[:, 1], return_index=True)[1]]
                        tp[x[:, 0], j] = True
            tps.append(tp); conf.append(s); pcls.append(k); tcls.append(t[:, 0])
    tp, cf, pc, tc = map(np.concatenate, (tps, conf, pcls, tcls))
    res = ap_per_class(tp, cf, pc, tc, names={i: n for i, n in enumerate(C)})
    p, rr, ap, u = res[2], res[3], res[5], res[6].astype(int)
    with open(OUT / f"{tag}.csv", "w", newline="") as fh:
        c = csv.writer(fh); c.writerow(["id", "class", "P", "R", "AP50", "AP50-95", "gt"])
        for i, k in enumerate(u):
            c.writerow([k, C[k], f"{p[i]:.3f}", f"{rr[i]:.3f}", f"{ap[i, 0]:.3f}", f"{ap[i].mean():.3f}", int((tc == k).sum())])
    print(f"BENCH {tag}: cls {len(u)} P {p.mean():.4f} R {rr.mean():.4f} mAP50 {ap[:, 0].mean():.4f} mAP50-95 {ap.mean():.4f} gt {len(tc)}", flush=True)
m = RTDETR(w) if "rtdetr" in TAG else YOLO(w)
print("BENCH", TAG, "GPU", torch.cuda.get_device_name(0), "params %.2f M" % (sum(x.numel() for x in m.model.parameters()) / 1e6), flush=True)
t = []
for k, (f, _) in enumerate(real[:320]):
    r = m.predict(f, imgsz=1280, conf=0.25, verbose=False, half=True)[0]
    if k >= 20: t.append(sum(r.speed.values()))
print(f"BENCH {TAG} latency {np.mean(t):.2f} ms, {1000 / np.mean(t):.1f} FPS", flush=True)
run(m, 1280, real, f"{TAG}__real_all"); run(m, 1280, unseen, f"{TAG}__real_unseen_by_old"); run(m, 1280, bed, f"{TAG}__bed_heldout")
EOF
