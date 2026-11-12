#!/bin/bash
#SBATCH --job-name=in_bench
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/bench_%j.log
# Paper benchmark: old YOLO26-L P2 vs YOLO11s v1 / v2, one scorer for all, predictions mapped to the new class list.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
python -u - <<'EOF'
import csv, glob, json, os
from pathlib import Path
import numpy as np, torch
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class

CM = json.load(open("class_map.json")); C = CM["classes"]
P = {26: 81, 27: 85, 28: 78, 95: 77, 65: 80, 5: 63, 29: 99}      # old class -> new class where the plain map has a tag
OLD = {}
for k, v in CM["old2new"].items():
    k = int(k); OLD[k] = P.get(k, v if isinstance(v, int) else None)
OLD_CAN = {v for v in OLD.values() if v is not None}
D = Path("india_train_v2"); OUT = Path("runs/bench"); OUT.mkdir(parents=True, exist_ok=True)
old_train = {Path(n).stem for n in os.listdir("/data/common_datasets/traffic_signs_india_split/images/train")}

def files(img_dir, lab_dir, keep=None):
    out = []
    for f in sorted(glob.glob(f"{img_dir}/*.jpg")):
        s = Path(f).stem
        if keep is None or keep(s):
            out.append((f, f"{lab_dir}/{s}.txt"))
    return out

real = files(D / "images/test", D / "labels/test")
real_unseen = files(D / "images/test", D / "labels/test", lambda s: s not in old_train)
bed_held = files(D / "images/test_bed", D / "labels/test_bed")
bed_all = [(f, f[:-4] + ".txt") for f in sorted(glob.glob("india_labeled/*/*.jpg"))]
print("real", len(real), "real_unseen_by_old", len(real_unseen), "bed_held", len(bed_held), "bed_all", len(bed_all), flush=True)

IOUV = np.linspace(0.5, 0.95, 10)
def iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None); inter = wh[..., 0] * wh[..., 1]
    A = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); B = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (A[:, None] + B[None, :] - inter + 1e-9)

def run(model, sz, remap, fl, tag):
    conf, pcls, tps, tcls = [], [], [], []
    for i in range(0, len(fl), 16):
        part = fl[i:i + 16]
        for (f, lab), r in zip(part, model.predict([p for p, _ in part], imgsz=sz, conf=0.001, iou=0.7, max_det=300, verbose=False, half=True)):
            H, W = r.orig_shape
            t = []
            if os.path.exists(lab):
                for l in open(lab):
                    q = l.split()
                    if len(q) >= 5 and int(q[0]) != 99:
                        c, cx, cy, w, h = int(q[0]), *map(float, q[1:5])
                        t.append([c, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
            t = np.array(t).reshape(-1, 5)
            b = r.boxes.xyxy.cpu().numpy(); s = r.boxes.conf.cpu().numpy(); k = r.boxes.cls.cpu().numpy().astype(int)
            if remap:
                k = np.array([-1 if OLD.get(int(x)) is None else OLD[int(x)] for x in k], dtype=int)
            m = (k >= 0) & (k != 99); b, s, k = b[m], s[m], k[m]
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
        w = csv.writer(fh); w.writerow(["id", "class", "P", "R", "AP50", "AP50-95", "gt"])
        for i, c in enumerate(u):
            w.writerow([c, C[c], f"{p[i]:.3f}", f"{rr[i]:.3f}", f"{ap[i, 0]:.3f}", f"{ap[i].mean():.3f}", int((tc == c).sum())])
    common = np.array([c in OLD_CAN for c in u])
    def line(mask):
        return f"cls {int(mask.sum())} P {p[mask].mean():.4f} R {rr[mask].mean():.4f} mAP50 {ap[mask, 0].mean():.4f} mAP50-95 {ap[mask].mean():.4f}"
    print(f"{tag}: ALL {line(np.ones(len(u), bool))} | COMMON {line(common)} | gt {len(tc)}", flush=True)

MODELS = [("p2_640", "../sandeep_india_check/p2_best.pt", 640, True), ("p2_1280", "../sandeep_india_check/p2_best.pt", 1280, True),
          ("v1_1280", "checkpoints/india_v1_best.pt", 1280, False), ("v2_1280", "checkpoints/india_v2_best.pt", 1280, False)]
print("GPU", torch.cuda.get_device_name(0), flush=True)
for tag, w, sz, remap in MODELS:
    if not os.path.exists(w):
        w = "/data/common_datasets/ai24mtech02003/sandeep_india_check/p2_best.pt"
    m = YOLO(w)
    n = sum(x.numel() for x in m.model.parameters())
    info = m.model.info(verbose=False, imgsz=sz)
    print(f"{tag}: params {n / 1e6:.2f} M, GFLOPs {info[-1] if info else 'na'}", flush=True)
    t = []
    for k, (f, _) in enumerate(real[:320]):
        r = m.predict(f, imgsz=sz, conf=0.25, verbose=False, half=True)[0]
        if k >= 20:
            t.append(sum(r.speed.values()))
    print(f"{tag}: latency mean {np.mean(t):.2f} ms (pre+inf+post, batch 1), {1000 / np.mean(t):.1f} FPS", flush=True)
    run(m, sz, remap, real, f"{tag}__real_all")
    run(m, sz, remap, real_unseen, f"{tag}__real_unseen_by_old")
    run(m, sz, remap, bed_held, f"{tag}__bed_heldout")
    if not tag.startswith("v2"):
        run(m, sz, remap, bed_all, f"{tag}__bed_all")
print("done")
EOF
