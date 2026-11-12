"""Review sheets with near-duplicate grouping. usage: python group_sheets.py <tag|class_id> <out_prefix> [margin]
Boxes of that tag (P / INFO) or new class id are cropped (with margin), grouped by crop dHash (Hamming <= 10)
and one crop per group is drawn with its group number -> <out_prefix>_NN.jpg + <out_prefix>_groups.json."""
import json, sys
from pathlib import Path
import cv2, numpy as np
R = Path(__file__).parent
tag = sys.argv[1]; out = sys.argv[2]; mg = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
want = (lambda v: v == tag) if not tag.isdigit() else (lambda v: v == int(tag))
items = []
for l in open(R / "inventory.jsonl"):
    r = json.loads(l)
    for k, b in enumerate(r.get("boxes", [])):
        if want(b[1]): items.append((r["path"], r["kind"], k, b))
def dh(g):
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    return int("".join("1" if a > b else "0" for a, b in zip(s[:, 1:].ravel(), s[:, :-1].ravel())), 2)
groups = []   # [hash, [members], tile]
for path, kind, k, (o, n, cx, cy, w, h) in items:
    im = cv2.imread(path); H, W = im.shape[:2]
    x0, y0, x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
    m = mg * max(x1 - x0, y1 - y0)
    X0, Y0, X1, Y1 = int(max(0, x0 - m)), int(max(0, y0 - m)), int(min(W, x1 + m)), int(min(H, y1 + m))
    cr = im[Y0:Y1, X0:X1].copy()
    cv2.rectangle(cr, (int(x0 - X0), int(y0 - Y0)), (int(x1 - X0), int(y1 - Y0)), (0, 255, 0), max(1, cr.shape[1] // 120))
    hh = dh(cv2.cvtColor(im[int(y0):int(max(y1, y0 + 2)), int(x0):int(max(x1, x0 + 2))], cv2.COLOR_BGR2GRAY))
    g = next((g for g in groups if bin(g[0] ^ hh).count("1") <= 10 and g[3] == o), None)
    if g is None:
        groups.append([hh, [], cv2.resize(cr, (150, 150)), o]); g = groups[-1]
    g[1].append([path, k])
tiles = []
for i, (hh, mem, t, o) in enumerate(groups):
    t = t.copy(); cv2.rectangle(t, (0, 0), (150, 18), (0, 0, 0), -1)
    cv2.putText(t, f"{i} n{len(mem)} o{o}", (2, 13), 0, 0.42, (0, 255, 255), 1); tiles.append(t)
tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 10)
for s in range(0, len(tiles), 80):
    part = tiles[s:s + 80]
    cv2.imwrite(f"{out}_{s // 80:02d}.jpg", np.vstack([np.hstack(part[i:i + 10]) for i in range(0, len(part), 10)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
json.dump([{"group": i, "old": o, "members": mem} for i, (hh, mem, t, o) in enumerate(groups)], open(f"{out}_groups.json", "w"))
print(len(items), "boxes", len(groups), "groups", (len(tiles) + 79) // 80, "sheets")
