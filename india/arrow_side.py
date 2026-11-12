"""Left/right test for blue turn-arrow signs: x-centroid of the white arrow pixels in the upper half of the crop
(negative = arrowhead on the left = left turn). usage: python arrow_side.py <class> [<class> ...] -> arrow_side.csv + sheet."""
import csv, json, sys, random
import cv2, numpy as np
want = set(int(c) for c in sys.argv[1:])
res = []
for l in open("inventory.jsonl"):
    r = json.loads(l)
    for k, b in enumerate(r.get("boxes", [])):
        if b[1] in want:
            im = cv2.imread(r["path"]); H, W = im.shape[:2]; _, n, cx, cy, w, h = b
            cr = im[int(max(0, (cy - h/2)*H)):int((cy + h/2)*H), int(max(0, (cx - w/2)*W)):int((cx + w/2)*W)]
            if cr.size == 0: continue
            g = cv2.resize(cr, (64, 64)); hsv = cv2.cvtColor(g, cv2.COLOR_BGR2HSV)
            white = (hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 150)
            yy, xx = np.mgrid[0:64, 0:64]; disc = (xx - 32) ** 2 + (yy - 32) ** 2 < 26 ** 2
            m = white & disc & (yy < 34)
            s = float((xx[m] - 32).mean() / 32) if m.sum() > 15 else 0.0
            res.append((r["path"], k, n, r["kind"], round(s, 3), g))
csv.writer(open("arrow_side.csv", "w", newline="")).writerows([(p, k, n, kd, s) for p, k, n, kd, s, _ in res])
rnd = random.Random(0); lines = []
for c in sorted(want):
    for lab, f in (("left", lambda s: s < -0.08), ("right", lambda s: s > 0.08), ("unsure", lambda s: -0.08 <= s <= 0.08)):
        sel = [x for x in res if x[2] == c and f(x[4])]; pick = rnd.sample(sel, min(20, len(sel)))
        t = [x[5] for x in pick] + [np.zeros((64, 64, 3), np.uint8)] * (20 - len(pick))
        hd = np.zeros((64, 140, 3), np.uint8); cv2.putText(hd, f"{c} {lab} {len(sel)}", (2, 38), 0, 0.5, (255, 255, 255), 1)
        lines.append(np.hstack([hd] + t))
cv2.imwrite("review/arrow_side.jpg", np.vstack(lines)); print(len(res))
