"""Evenly spaced raw frames of the test videos (for seeing the signs actually present) + detection counts."""
import csv, collections, sys, cv2, numpy as np
for v, step_s in (("cam_20261106_100001", 12), ("cam_20261106_100849", 12)):
    rows = list(csv.DictReader(open(f"video_test_v1/{v}_detections.csv")))
    c = collections.Counter(r["class_name"] for r in rows)
    print(v, len(rows), "boxes", len(c), "classes"); print("  ", ", ".join(f"{k} {n}" for k, n in c.most_common()))
    cap = cv2.VideoCapture(f"infer_video/{v}.mp4"); n = int(cap.get(7)); fps = cap.get(5); tiles = []
    for f in range(0, n, int(step_s * fps)):
        cap.set(1, f); ok, im = cap.read()
        if not ok: break
        t = cv2.resize(im, (480, 300)); cv2.rectangle(t, (0, 0), (70, 18), (0, 0, 0), -1)
        cv2.putText(t, f"{f / fps:.0f}s", (2, 14), 0, 0.5, (0, 255, 255), 1); tiles.append(t)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
    for s in range(0, len(tiles), 16):
        p = tiles[s:s + 16]; cv2.imwrite(f"review/vid_{v[-6:]}_{s // 16}.jpg", np.vstack([np.hstack(p[i:i + 4]) for i in range(0, len(p), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
