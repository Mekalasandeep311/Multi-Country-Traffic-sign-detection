#!/bin/bash
#SBATCH --job-name=in_rescore
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rescore_%j.log
# Final paper scoring: every model, one H200, COCO-style matching (highest confidence first),
# bootstrap 95% CI over images, latency + GFLOPs. Results -> runs/rescore/
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
python -u - <<'EOF'
import csv, glob, json, os, time
from pathlib import Path
import numpy as np, torch
from ultralytics import YOLO, RTDETR
from ultralytics.utils.metrics import ap_per_class

CM = json.load(open("class_map.json")); C = CM["classes"]
P = {26: 81, 27: 85, 28: 78, 95: 77, 65: 80, 5: 63, 29: 99}
OLD = {int(k): P.get(int(k), v if isinstance(v, int) else None) for k, v in CM["old2new"].items()}
D = Path("india_train_v2"); OUT = Path("runs/rescore"); OUT.mkdir(parents=True, exist_ok=True)
old_train = {Path(n).stem for n in os.listdir("/data/common_datasets/traffic_signs_india_split/images/train")}
def files(i, l, keep=None):
    return [(f, f"{l}/{Path(f).stem}.txt") for f in sorted(glob.glob(f"{i}/*.jpg")) if keep is None or keep(Path(f).stem)]
SETS = {"real_all": files(D / "images/test", D / "labels/test"),
        "real_unseen_by_old": files(D / "images/test", D / "labels/test", lambda s: s not in old_train),
        "bed_heldout": files(D / "images/test_bed", D / "labels/test_bed"),
        "bed_all": [(f, f[:-4] + ".txt") for f in sorted(glob.glob("india_labeled/*/*.jpg"))]}
print({k: len(v) for k, v in SETS.items()}, flush=True)

def bw(tag):
    return (glob.glob(f"runs/detect/runs/bench_{tag}/weights/best.pt") + glob.glob(f"runs/bench_{tag}/weights/best.pt"))[0]
OLDW = "../sandeep_india_check/p2_best.pt"
if not os.path.exists(OLDW): OLDW = "/data/common_datasets/ai24mtech02003/sandeep_india_check/p2_best.pt"
# tag, weights, imgsz, remap old ids, sets (bed_all only for models that never saw test-bed frames)
MODELS = [("old_yolo26l_p2_640", OLDW, 640, True, ["real_unseen_by_old", "bed_heldout", "bed_all"]),
          ("old_yolo26l_p2_1280", OLDW, 1280, True, ["real_unseen_by_old", "bed_heldout", "bed_all"]),
          ("ours_v1", "checkpoints/india_v1_best.pt", 1280, False, ["real_all", "real_unseen_by_old", "bed_heldout", "bed_all"]),
          ("ours_v2", "checkpoints/india_v2_best.pt", 1280, False, ["real_all", "real_unseen_by_old", "bed_heldout"]),
          ("ours_v3", "checkpoints/india_v3_best.pt", 1280, False, ["real_all", "real_unseen_by_old", "bed_heldout"])]
for t in ["yolov8s", "yolov10s", "yolo11s", "yolo11s_p2", "yolo26l", "yolo26l_p2", "rtdetr_l"]:
    MODELS.append((f"bench_{t}", bw(t), 1280, False, ["real_all", "real_unseen_by_old", "bed_heldout"]))

IOUV = np.linspace(0.5, 0.95, 10)
def iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None); inter = wh[..., 0] * wh[..., 1]
    A = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); B = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (A[:, None] + B[None, :] - inter + 1e-9)

def match(b, s, k, t):
    """COCO-style: predictions in descending confidence each take the best still-free GT of the same class."""
    tp = np.zeros((len(b), 10), bool)
    if not len(b) or not len(t): return tp
    I = iou(b, t[:, 1:]) * (k[:, None] == t[:, 0][None, :])
    order = np.argsort(-s, kind="stable")
    cand = [i for i in order if I[i].max() >= 0.5]
    for j, th in enumerate(IOUV):
        free = np.ones(len(t), bool)
        for i in cand:
            row = np.where(free, I[i], 0)
            g = row.argmax()
            if row[g] >= th: tp[i, j] = True; free[g] = False
    return tp

def summarise(tp, cf, pc, tc):
    res = ap_per_class(tp, cf, pc, tc, names={i: n for i, n in enumerate(C)})
    return res[2], res[3], res[5], res[6].astype(int)

def evaluate(model, sz, remap, fl, tag, B=1000):
    per = []   # per image: (tp, conf, pcls, tcls)
    for i in range(0, len(fl), 8):
        part = fl[i:i + 8]
        for (f, lab), r in zip(part, model.predict([p for p, _ in part], imgsz=sz, conf=0.001, iou=0.7, max_det=300, verbose=False, half=True)):
            H, W = r.orig_shape; t = []
            if os.path.exists(lab):
                for l in open(lab):
                    q = l.split()
                    if len(q) >= 5 and int(q[0]) != 99:
                        c, cx, cy, w, h = int(q[0]), *map(float, q[1:5])
                        t.append([c, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
            t = np.array(t).reshape(-1, 5)
            b = r.boxes.xyxy.cpu().numpy(); s = r.boxes.conf.float().cpu().numpy(); k = r.boxes.cls.cpu().numpy().astype(int)
            if remap: k = np.array([-1 if OLD.get(int(x)) is None else OLD[int(x)] for x in k], dtype=int)
            m = (k >= 0) & (k != 99); b, s, k = b[m], s[m], k[m]
            per.append((match(b, s, k, t), s, k, t[:, 0]))
    cat = lambda idx: [np.concatenate([per[j][q] for j in idx]) for q in range(4)]
    p, rr, ap, u = summarise(*cat(range(len(per))))
    with open(OUT / f"{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "class", "P", "R", "AP50", "AP50-95", "gt"])
        tc = cat(range(len(per)))[3]
        for i, c in enumerate(u):
            w.writerow([c, C[c], f"{p[i]:.3f}", f"{rr[i]:.3f}", f"{ap[i, 0]:.3f}", f"{ap[i].mean():.3f}", int((tc == c).sum())])
    rng = np.random.default_rng(0); bs50, bs95 = [], []
    for _ in range(B):
        idx = rng.integers(0, len(per), len(per))
        _, _, a, _ = summarise(*cat(idx)); bs50.append(a[:, 0].mean()); bs95.append(a.mean())
    lo50, hi50 = np.percentile(bs50, [2.5, 97.5]); lo95, hi95 = np.percentile(bs95, [2.5, 97.5])
    row = dict(model=tag.split("__")[0], set=tag.split("__")[1], images=len(per), classes=len(u), gt=len(tc),
               P=round(p.mean(), 4), R=round(rr.mean(), 4), mAP50=round(ap[:, 0].mean(), 4), mAP50_lo=round(lo50, 4), mAP50_hi=round(hi50, 4),
               mAP5095=round(ap.mean(), 4), mAP5095_lo=round(lo95, 4), mAP5095_hi=round(hi95, 4))
    print("RESCORE", json.dumps(row), flush=True)
    return row

GPU = torch.cuda.get_device_name(0); print("GPU", GPU, flush=True)
rows, speed = [], []
for tag, w, sz, remap, sets in MODELS:
    m = RTDETR(w) if "rtdetr" in tag else YOLO(w)
    n = sum(x.numel() for x in m.model.parameters()) / 1e6
    try: gf = m.model.info(verbose=False, imgsz=sz)[-1]
    except Exception: gf = float("nan")
    lat = []
    for k, (f, _) in enumerate(SETS["real_all"][:320]):
        r = m.predict(f, imgsz=sz, conf=0.25, verbose=False, half=True)[0]
        if k >= 20: lat.append((r.speed["preprocess"], r.speed["inference"], r.speed["postprocess"]))
    lat = np.array(lat); tot = lat.sum(1)
    sp = dict(model=tag, gpu=GPU, imgsz=sz, params_M=round(n, 2), GFLOPs=round(float(gf), 1) if gf == gf else "", infer_ms=round(lat[:, 1].mean(), 2),
              total_ms=round(tot.mean(), 2), total_p95_ms=round(np.percentile(tot, 95), 2), FPS=round(1000 / tot.mean(), 1))
    print("SPEED", json.dumps(sp), flush=True); speed.append(sp)
    for s in sets: rows.append(evaluate(m, sz, remap, SETS[s], f"{tag}__{s}"))
    del m; torch.cuda.empty_cache()
for name, data in (("accuracy.csv", rows), ("speed.csv", speed)):
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
print("done", flush=True)
EOF
