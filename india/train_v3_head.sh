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
