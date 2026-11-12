"""Spot-check sheets: 20 random crops per class (r = real photo, w = web), 4 classes per sheet -> review/spot/s_NN.jpg."""
import json, random, collections
from pathlib import Path
import cv2, numpy as np
R = Path(__file__).parent
C = json.load(open(R / "class_map.json"))["classes"]
by = collections.defaultdict(list)
for l in open(R / "inventory.jsonl"):
    r = json.loads(l)
    for k, b in enumerate(r.get("boxes", [])):
        if isinstance(b[1], int) and b[1] < 98: by[b[1]].append((r["path"], r["kind"], b))
rnd = random.Random(3); out = R / "review" / "spot"; out.mkdir(parents=True, exist_ok=True)
cls = [c for c in range(98) if by[c]]
rows = []
for c in cls:
    pick = rnd.sample(by[c], min(20, len(by[c])))
    tiles = []
    for path, kind, (o, n, cx, cy, w, h) in pick:
        im = cv2.imread(path); H, W = im.shape[:2]
        x0, y0, x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
        m = 0.25 * max(x1 - x0, y1 - y0)
        cr = im[int(max(0, y0 - m)):int(min(H, y1 + m)), int(max(0, x0 - m)):int(min(W, x1 + m))]
        t = cv2.resize(cr, (110, 110)); cv2.putText(t, kind[0], (3, 14), 0, 0.5, (0, 255, 255), 2); tiles.append(t)
    tiles += [np.zeros((110, 110, 3), np.uint8)] * (20 - len(tiles))
    lab = np.zeros((220, 150, 3), np.uint8)
    for i, part in enumerate([f"{c}", C[c][:16], C[c][16:32], f"n={len(by[c])}"]):
        cv2.putText(lab, part, (4, 30 + 40 * i), 0, 0.5 if i else 1.0, (255, 255, 255), 1 if i else 2)
    rows.append(np.hstack([lab, np.vstack([np.hstack(tiles[:10]), np.hstack(tiles[10:])])]))
for s in range(0, len(rows), 4):
    cv2.imwrite(str(out / f"s_{s // 4:02d}.jpg"), np.vstack(rows[s:s + 4]), [cv2.IMWRITE_JPEG_QUALITY, 82])
print(len(cls), "classes", (len(rows) + 3) // 4, "sheets")
