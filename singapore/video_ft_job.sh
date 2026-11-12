#!/bin/bash
#SBATCH --job-name=sg_vidft
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/video_ft_%j.log

# Fine-tune best.pt on full training set + reviewed video tiles (positives + hard negatives),
# then compare old vs new on the real val set and on the held-out last 20% of the video.
set -uo pipefail
P=/data/common_datasets/ai24mtech02003/sandeep_singapore
cd $P
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/common_datasets/ai24mtech02003/envs/sg_signs

python - <<'EOF'
import glob, yaml
base = sorted(glob.glob("sg_signs_train/images/train/*.jpg"))
vid = sorted(glob.glob("video_ft/images/train/*.jpg"))
import os
lst = [os.path.abspath(p) for p in base]
for r in range(3):                       # repeat video tiles 3x via hard links (list de-dup safe)
    for p in vid:
        q = p if r == 0 else p.replace(".jpg", f"_r{r}.jpg")
        if r and not os.path.exists(q):
            os.link(p, q)
            os.link(p.replace("/images/", "/labels/").replace(".jpg", ".txt"),
                    q.replace("/images/", "/labels/").replace(".jpg", ".txt"))
        lst.append(os.path.abspath(q))
open("video_ft/ft_train.txt", "w").write("\n".join(lst) + "\n")
d = yaml.safe_load(open("sg_signs_train/data.yaml"))
d["train"] = os.path.abspath("video_ft/ft_train.txt")
d["val"] = os.path.abspath("sg_signs_train/images/val")
yaml.safe_dump(d, open("video_ft/ft_data.yaml", "w"), sort_keys=False, allow_unicode=True)
print(f"train list: {len(base)} base tiles + {3 * len(vid)} video tiles")
EOF

python -u - <<EOF
from ultralytics import YOLO
YOLO("runs/sg_signs_v2/weights/best.pt").train(
    data="video_ft/ft_data.yaml", project="$P/runs", name="sg_signs_v3_video", exist_ok=True,
    epochs=20, imgsz=640, batch=64, device=0, workers=8,
    optimizer="SGD", lr0=0.001, lrf=0.2, momentum=0.937, weight_decay=5e-4, warmup_epochs=0, cos_lr=True,
    patience=8, close_mosaic=5, mosaic=1.0, mixup=0.1, hsv_v=0.5, scale=0.5, translate=0.1,
    fliplr=0.0, flipud=0.0, degrees=0.0)
EOF

python -u - <<'EOF'
from ultralytics import YOLO
for tag, w in (("OLD v2", "runs/sg_signs_v2/weights/best.pt"), ("NEW v3", "runs/sg_signs_v3_video/weights/best.pt")):
    m = YOLO(w)
    a = m.val(data="sg_signs_train/data.yaml", imgsz=640, batch=32, plots=False, verbose=False, project="runs/v3_eval", name="a", exist_ok=True)
    b = m.val(data="video_ft/data_video_val.yaml", imgsz=1280, batch=8, plots=False, verbose=False, conf=0.25, project="runs/v3_eval", name="b", exist_ok=True)
    print(f"{tag}: real val tiles mAP50 {a.box.map50:.3f} mAP50-95 {a.box.map:.3f} R {a.box.mr:.3f} | "
          f"held-out video: P {b.box.mp:.3f} R {b.box.mr:.3f} mAP50 {b.box.map50:.3f}")
EOF
