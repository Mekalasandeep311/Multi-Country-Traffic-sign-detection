"""Run the trained YOLO sign detector on images or a video and draw
class name, confidence, estimated distance and an FPS/latency header.

False-positive control:
  * CONF for ordinary signs; SYNTH_CONF for classes trained (almost) only on synthetic images
    (listed in synthetic_only_classes.txt) and for big electronic / panel classes that tend to
    fire on hoardings, bins and building facades.
  * Video only: signs are tracked (ByteTrack) and drawn only once a track has been seen in
    MIN_HITS frames; the label is the track's confidence-weighted majority class.

usage: python detect.py [source] [weights] [out_dir]
       source: image file, folder of .jpg/.png, or a video file (default: this folder)
       weights: default runs/sg_signs_v2/weights/best.pt
       out_dir: images -> output/yolo_preview, videos -> next to the video
"""
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from ultralytics import YOLO

from autolabel import HFOV_DEG, ROOT, SIGN_HEIGHT_M, color_for

CONF = 0.50
SYNTH_CONF = 0.70
SYNTH_LIST = ROOT / "synthetic_only_classes.txt"
PANEL_KEYWORDS = ("Display", "RVMS", "Gantry", "Green Arrow", "Red Cross", "Flashing Amber",
                  "Directional", "Intermediate Sign", "Road Closure")
TRACK_CONF = 0.25      # lower bound handed to the tracker so it can keep tracks alive
MIN_HITS = 3           # frames a track must be seen in before it is drawn


def class_thresholds(names):
    synth = set(SYNTH_LIST.read_text().splitlines()) if SYNTH_LIST.exists() else set()
    return {k: SYNTH_CONF if (n in synth or any(w in n for w in PANEL_KEYWORDS)) else CONF
            for k, n in names.items()}


def annotate(frame, dets, names, tag, fps, latency_ms):
    """dets: list of (x0, y0, x1, y1, conf, cls)."""
    H, W = frame.shape[:2]
    f_px = (W / 2) / math.tan(math.radians(HFOV_DEG / 2))
    for x0, y0, x1, y1, c, k in dets:
        x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
        col = color_for(int(k) + 1)
        dist = f_px * SIGN_HEIGHT_M / max(1, y1 - y0)
        text = f"{names[int(k)]} {c:.2f} {dist:.1f}m"
        cv2.rectangle(frame, (x0, y0), (x1, y1), col, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = y0 - 4 if y0 - th - 8 > 0 else y1 + th + 6
        cv2.rectangle(frame, (x0, ty - th - 4), (x0 + tw + 4, ty + 3), col, -1)
        cv2.putText(frame, text, (x0 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    header = f"[{tag}] FPS: {fps:.1f}  Latency: {latency_ms:.1f} ms  Det: {len(dets)}"
    cv2.putText(frame, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


def imgsz_for(frame):
    # Run at (close to) native resolution: the model was trained on native-scale tiles.
    return int(math.ceil(max(frame.shape[:2]) / 32) * 32)


def predict(model, frame, thr):
    t = time.time()
    r = model.predict(frame, imgsz=imgsz_for(frame), conf=min(thr.values()), verbose=False)[0]
    dets = [(*b, c, int(k)) for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())
            if c >= thr[int(k)]]
    return dets, (time.time() - t) * 1000


class TrackFilter:
    """Keeps per-track hit counts and class votes; returns only confirmed, confident tracks."""

    def __init__(self, thr):
        self.thr = thr
        self.hits = Counter()
        self.votes = defaultdict(Counter)
        self.confs = defaultdict(list)

    def __call__(self, result):
        b = result.boxes
        if b.id is None:
            return []
        out = []
        for box, c, k, tid in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist(), b.id.int().tolist()):
            self.hits[tid] += 1
            self.votes[tid][int(k)] += c
            self.confs[tid] = (self.confs[tid] + [c])[-10:]
            cls = self.votes[tid].most_common(1)[0][0]
            mean_c = sum(self.confs[tid]) / len(self.confs[tid])
            if self.hits[tid] >= MIN_HITS and mean_c >= self.thr[cls]:
                out.append((*box, c, cls))
        return out


def detect_video(model, names, tag, source: Path, out_dir: Path):
    """Write <stem>_detected.mp4 (every frame annotated), <stem>_detected_frames/ (only frames
    with at least one sign) and <stem>_detections.csv (one row per box)."""
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 0:
        sys.exit(f"cannot read {source} (incomplete upload or unsupported codec?)")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_dir = out_dir / f"{source.stem}_detected_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{source.stem}_detected.mp4"
    csv = open(out_dir / f"{source.stem}_detections.csv", "w")
    csv.write("frame,time_s,class_id,class_name,conf,x0,y0,x1,y1\n")

    track_filter = TrackFilter(class_thresholds(names))
    writer, idx, saved, t_start = None, 0, 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        r = model.track(frame, imgsz=imgsz_for(frame), conf=TRACK_CONF, persist=True,
                        tracker="bytetrack.yaml", verbose=False)[0]
        dets = track_filter(r)
        ms = (time.time() - t0) * 1000
        frame = annotate(frame, dets, names, tag, 1000 / ms, ms)
        if writer is None:
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps_in, frame.shape[1::-1])
        writer.write(frame)

        if dets:
            t = idx / fps_in
            classes = sorted({names[k] for *_, k in dets})
            label = "_".join(c.replace(" ", "-").replace("/", "-") for c in classes)[:120]
            cv2.imwrite(str(frames_dir / f"f{idx:06d}_{int(t // 60):02d}m{t % 60:05.2f}s_{label}.jpg"), frame)
            saved += 1
            for x0, y0, x1, y1, c, k in dets:
                csv.write(f'{idx},{t:.2f},{k},"{names[k]}",{c:.3f},{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}\n')
        idx += 1
        if idx % 500 == 0:
            el = time.time() - t_start
            print(f"{idx}/{total} frames  {idx / el:.1f} fps  frames with signs: {saved}", flush=True)

    cap.release()
    csv.close()
    if writer:
        writer.release()
    print(f"done: {idx} frames, {saved} with detections")
    print("saved", video_path, frames_dir, csv.name)


def main(source: Path, weights: Path, out_dir_override: Path = None):
    model = YOLO(str(weights))
    names = model.names
    tag = weights.stem

    if source.is_file() and source.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        detect_video(model, names, tag, source, out_dir_override or source.parent)
        return

    out_dir = out_dir_override or ROOT / "output" / "yolo_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    thr = class_thresholds(names)
    files = [source] if source.is_file() else sorted(
        p for p in source.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for p in files:
        frame = cv2.imread(str(p))
        dets, ms = predict(model, frame, thr)
        cv2.imwrite(str(out_dir / p.name), annotate(frame, dets, names, tag, 1000 / ms, ms))
        print(f"{p.name}: {ms:.0f} ms  [{', '.join(f'{names[k]} {c:.2f}' for *_, c, k in dets)}]")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT,
         Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "runs" / "sg_signs_v2" / "weights" / "best.pt",
         Path(sys.argv[3]) if len(sys.argv) > 3 else None)
