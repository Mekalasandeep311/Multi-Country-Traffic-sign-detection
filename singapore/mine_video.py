"""Collect detections from a video for review (hard-negative / extra-positive mining).

Samples one frame every STEP frames, runs the model at a low threshold, and writes (video_mine/<stem>/):
  frames/<frame>.jpg          sampled frames that have at least one detection
  detections.csv              id, frame, time_s, class, conf, box
  sheets/<class>.jpg          numbered crops per predicted class (highest conf first) for review

After review, the user lists wrong detection ids (false positives) -> hard negatives, and
correct ones -> extra real positives.

usage: python mine_video.py <video> [weights] [step]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent
CONF = 0.20
PER_SHEET = 60


def main(video, weights, step):
    out = ROOT / "video_mine" / video.stem
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "sheets").mkdir(exist_ok=True)
    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows, idx = [], 0
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            _, frame = cap.retrieve()
            r = model.predict(frame, imgsz=int(np.ceil(max(frame.shape[:2]) / 32) * 32), conf=CONF, verbose=False)[0]
            if len(r.boxes):
                cv2.imwrite(str(out / "frames" / f"{idx:06d}.jpg"), frame)
                for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                    rows.append([len(rows), idx, round(idx / fps, 2), model.names[int(k)], round(c, 3),
                                 *[round(v, 1) for v in b]])
        idx += 1
        if idx % (step * 500) == 0:
            print(f"{idx}/{total} frames, {len(rows)} detections", flush=True)
    cap.release()

    with open(out / "detections.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "frame", "time_s", "class", "conf", "x0", "y0", "x1", "y1"])
        w.writerows(rows)

    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r[3]].append(r)
    for cls, rs in by_cls.items():
        rs = sorted(rs, key=lambda r: -r[4])
        if len(rs) > PER_SHEET:            # top 30 + an even spread of the rest (lower confidences)
            rs = rs[:30] + [rs[i] for i in np.linspace(30, len(rs) - 1, PER_SHEET - 30).astype(int)]
        tiles = []
        for i, fr, t, c, conf, x0, y0, x1, y1 in rs:
            im = cv2.imread(str(out / "frames" / f"{fr:06d}.jpg"))
            m = 1.2 * max(x1 - x0, y1 - y0) + 10
            ox, oy = int(max(0, x0 - m)), int(max(0, y0 - m))
            crop = im[oy:int(y1 + m), ox:int(x1 + m)].copy()
            cv2.rectangle(crop, (int(x0) - ox, int(y0) - oy), (int(x1) - ox, int(y1) - oy), (0, 0, 255), 1)
            crop = cv2.resize(crop, (150, 150))
            cv2.rectangle(crop, (0, 0), (150, 16), (0, 0, 0), -1)
            cv2.putText(crop, f"#{i}  {conf:.2f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            tiles.append(crop)
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 10)
        sheet = np.vstack([np.hstack(tiles[k:k + 10]) for k in range(0, len(tiles), 10)])
        head = np.zeros((30, sheet.shape[1], 3), np.uint8)
        cv2.putText(head, f"{cls}  ({len(by_cls[cls])} detections)", (5, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        safe = cls.replace("/", "-")
        cv2.imwrite(str(out / "sheets" / f"{len(by_cls[cls]):05d}_{safe}.jpg"), np.vstack([head, sheet]))
    print(f"done: {idx} frames read, {len(rows)} detections in {len(by_cls)} classes -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]),
         Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "runs" / "sg_signs_v2" / "weights" / "best.pt",
         int(sys.argv[3]) if len(sys.argv) > 3 else 15)
