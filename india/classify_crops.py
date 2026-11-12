"""Run the label-checking classifier on every labelled box -> check.csv (pred class, mirrored?, prob)."""
import csv, json, sys
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO
R = Path(__file__).parent
m = YOLO(sys.argv[1])
names = m.names
items = []
for l in open(R / "inventory.jsonl"):
    r = json.loads(l)
    for k, b in enumerate(r.get("boxes", [])):
        if b[1] is not None: items.append((r, k, b))
out = csv.writer(open(R / "check.csv", "w", newline=""))
out.writerow(["stem", "path", "kind", "box", "old", "new", "pred", "pred_prob", "own_prob", "own_mirror_prob", "top3"])
batch = []
def flush():
    res = m.predict([c for c, _ in batch], imgsz=128, verbose=False, half=True)
    for (c, (r, k, b)), p in zip(batch, res):
        pr = p.probs.data.cpu().numpy(); top = pr.argsort()[::-1][:3]
        n = b[1]
        own = mir = ""
        if isinstance(n, int) and n < 98:
            ci = [j for j, s in names.items() if s == f"c{n:03d}"]; mi = [j for j, s in names.items() if s == f"m{n:03d}"]
            own = f"{pr[ci[0]]:.3f}" if ci else ""; mir = f"{pr[mi[0]]:.3f}" if mi else ""
        out.writerow([r["stem"], r["path"], r["kind"], k, b[0], n, names[int(top[0])], f"{pr[top[0]]:.3f}", own, mir,
                      " ".join(f"{names[int(t)]}:{pr[t]:.2f}" for t in top)])
    batch.clear()
last = None; im = None
for r, k, b in items:
    if r["path"] != last:
        im = cv2.imread(r["path"]); last = r["path"]
    H, W = im.shape[:2]; _, _, cx, cy, w, h = b
    mx, my = 0.08 * w, 0.08 * h
    x0, y0 = int(max(0, (cx - w / 2 - mx) * W)), int(max(0, (cy - h / 2 - my) * H))
    x1, y1 = int(min(W, (cx + w / 2 + mx) * W)), int(min(H, (cy + h / 2 + my) * H))
    if x1 - x0 < 2 or y1 - y0 < 2: continue
    batch.append((cv2.resize(im[y0:y1, x0:x1], (128, 128)), (r, k, b)))
    if len(batch) == 256: flush()
if batch: flush()
print("done", len(items))
