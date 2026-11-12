#!/bin/bash
#SBATCH --job-name=in_v3
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_v3_%j.log
# India v3: fine-tune from read-only v2, only for No parking: 300 sign-swap images (test-bed + rule-book template) + real No parking frames x3.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs
export TAG=v3
python -u - <<'PY'
import json, os, random
from pathlib import Path
import cv2, numpy as np
import build_standard as B
from build_v2_data import gen
v = "cam_20261106_100849"
t = {x["id"]: x for x in json.load(open(f"india_mine/{v}/tracks.json"))}[859]
ok = [(f, b) for f, b in t["boxes"] if b[0] > 4 and b[1] > 4 and b[2] < 1916 and b[3] < 1196]
f, b = max(ok, key=lambda fb: (fb[1][2] - fb[1][0]) * (fb[1][3] - fb[1][1]))
im = cv2.imread(f"india_mine/{v}/frames/f{f:06d}.jpg"); x0, y0, x1, y1 = map(int, b[:4]); c = im[y0:y1, x0:x1].copy()
m = np.zeros(c.shape[:2], np.uint8); h, w = m.shape; cv2.ellipse(m, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1); m[0, 0] = 0
d = Path("templates_extra/013_No parking testbed"); d.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(d / "handbook.png"), np.dstack([c, m]))
cv2.imwrite("review/noparking_template.jpg", c)
tm = [B.cutout(d / "handbook.png"), B.T[13]]
folder = next(p.name for p in B.OUT.glob("013_*"))
print(gen((13, tm, 300, "syn_np_", folder)), flush=True)
D = Path("india_train_v2"); (D / "images/train_np").mkdir(exist_ok=True); (D / "labels/train_np").mkdir(exist_ok=True)
n = 0
for im in (B.OUT / folder).glob("syn_np_*.jpg"):
    os.symlink(os.path.realpath(im), D / "images/train_np" / im.name); (D / "labels/train_np" / (im.stem + ".txt")).write_text(im.with_suffix(".txt").read_text()); n += 1
for lab in (D / "labels/train").glob("cam_*.txt"):
    txt = lab.read_text()
    if any(l.split()[0] == "13" for l in txt.splitlines()):
        for k in range(3):
            os.symlink(os.path.realpath(D / "images/train" / (lab.stem + ".jpg")), D / "images/train_np" / f"{lab.stem}_x{k}.jpg"); (D / "labels/train_np" / f"{lab.stem}_x{k}.txt").write_text(txt); n += 1
print("extra train images", n, flush=True)
y = (D / "data.yaml").read_text().replace("train: images/train\n", "train: [images/train, images/train_np]\n"); (D / "data_v3.yaml").write_text(y)
PY
yolo detect train data=india_train_v2/data_v3.yaml model=checkpoints/india_v2_best.pt imgsz=1280 epochs=15 patience=6 batch=32 lr0=0.001 lrf=0.1 cos_lr=True warmup_epochs=1 optimizer=SGD \
     fliplr=0 flipud=0 degrees=0 mosaic=1.0 close_mosaic=3 mixup=0.1 workers=16 project=runs name=bench_v3 exist_ok=True plots=True
B=$(ls runs/detect/runs/bench_v3/weights/best.pt runs/bench_v3/weights/best.pt 2>/dev/null | head -1)
cp -n "$B" checkpoints/india_v3_best.pt; chmod 444 checkpoints/india_v3_best.pt; md5sum checkpoints/india_v3_best.pt
mkdir -p video_test_v3
for v in cam_20261106_100001 cam_20261106_100849; do
  python -u detect.py infer_video/$v.mp4 checkpoints/india_v3_best.pt video_test_v3
  python -u video_sheet.py video_test_v3/${v}_detections.csv infer_video/$v.mp4 video_test_v3/${v}_sheet
done
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
