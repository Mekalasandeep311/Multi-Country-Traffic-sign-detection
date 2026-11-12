"""Crops of one class split by the label checker's prediction. usage: python pred_sheet.py <class> <out.jpg> <predA> <predB>"""
import csv, sys, random
import cv2, numpy as np
c, out, A, B = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
rows = [r for r in csv.DictReader(open("check.csv")) if r["new"] == c]
rnd = random.Random(1); lines = []
for p in (A, B):
    sel = [r for r in rows if r["pred"] == p]; pick = rnd.sample(sel, min(20, len(sel)))
    tiles = []
    for r in pick:
        im = cv2.imread(r["path"]); H, W = im.shape[:2]
        txt = open(r["path"].rsplit(".", 1)[0] + ".txt").read().split("\n")[int(r["box"])].split()
        cx, cy, w, h = map(float, txt[1:5]); x0, y0, x1, y1 = (cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H
        t = cv2.resize(im[int(max(0,y0)):int(y1), int(max(0,x0)):int(x1)], (100, 100)); tiles.append(t)
    tiles += [np.zeros((100, 100, 3), np.uint8)] * (20 - len(tiles))
    lab = np.zeros((200, 160, 3), np.uint8); cv2.putText(lab, f"{p} n={len(sel)}", (4, 100), 0, 0.6, (255, 255, 255), 2)
    lines.append(np.hstack([lab, np.vstack([np.hstack(tiles[:10]), np.hstack(tiles[10:])])]))
cv2.imwrite(out, np.vstack(lines)); print(len(rows), {p: sum(r["pred"] == p for r in rows) for p in (A, B)})
