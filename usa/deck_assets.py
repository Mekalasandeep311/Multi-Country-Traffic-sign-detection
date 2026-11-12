"""Assets for the final USA deck: random 40-box sheet of v4 video boxes, and v3-vs-v4 frames of the same moment."""
import csv, random, collections
import cv2, numpy as np
OUT = "deck_assets"
import os; os.makedirs(OUT, exist_ok=True)
rows = list(csv.DictReader(open("video_test_v4/clip2_detections.csv")))
random.seed(0)
pick = random.sample(rows, 40)
cap = cv2.VideoCapture("infer_video/clip2.mp4")
tiles = []
for n, r in enumerate(pick):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame"])); ok, im = cap.read()
    x0, y0, x1, y1 = map(float, (r["x0"], r["y0"], r["x1"], r["y1"]))
    m = 1.2 * max(x1 - x0, y1 - y0) + 10
    ox, oy = int(max(0, x0 - m)), int(max(0, y0 - m))
    cr = im[oy:int(y1 + m), ox:int(x1 + m)].copy()
    cv2.rectangle(cr, (int(x0) - ox, int(y0) - oy), (int(x1) - ox, int(y1) - oy), (0, 255, 0), 1)
    cr = cv2.resize(cr, (220, 220))
    cv2.rectangle(cr, (0, 0), (220, 30), (0, 0, 0), -1)
    cv2.putText(cr, f"{n} {r['class_name'][:30]}", (2, 12), 0, 0.36, (255, 255, 255), 1)
    cv2.putText(cr, f"{float(r['conf']):.2f} t={float(r['time_s']):.0f}s", (2, 26), 0, 0.36, (0, 255, 255), 1)
    tiles.append(cr)
cv2.imwrite(f"{OUT}/v4_rand40.jpg", np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, 40, 8)]))
# same-moment comparison: v4 boxes of chosen classes vs whatever v3 drew in that frame
v3 = collections.defaultdict(list)
for r in csv.DictReader(open("video_test_v3/clip2_detections.csv")):
    v3[int(r["frame"])].append(r)
v4 = collections.defaultdict(list)
for r in rows:
    v4[int(r["frame"])].append(r)
def draw(im, rs, col):
    for r in rs:
        x0, y0, x1, y1 = (int(float(r[k])) for k in ("x0", "y0", "x1", "y1"))
        cv2.rectangle(im, (x0, y0), (x1, y1), col, 3)
        t = f"{r['class_name'][:34]} {float(r['conf']):.2f}"
        (tw, th), _ = cv2.getTextSize(t, 0, 0.6, 2)
        ty = max(th + 4, y0 - 4)
        cv2.rectangle(im, (x0, ty - th - 4), (x0 + tw + 4, ty + 4), col, -1)
        cv2.putText(im, t, (x0 + 2, ty), 0, 0.6, (0, 0, 0), 2)
want = ["No standing, bus stop with bus route, NY", "No standing with time restrictions, NY (Farmers Market)",
        "Turn only lanes", "No right turn", "One way (horizontal sign)", "Do not enter"]
used = []
for k in want:
    cand = [r for r in rows if r["class_name"] == k and all(abs(int(r["frame"]) - u) > 600 for u in used)]
    if not cand:
        print("none", k); continue
    r = max(cand, key=lambda r: float(r["conf"]) * min(1.0, (float(r["x1"]) - float(r["x0"])) / 60))
    f = int(r["frame"]); used.append(f)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
    a, b = im.copy(), im.copy()
    draw(a, v3.get(f, []), (0, 165, 255)); draw(b, v4[f], (0, 255, 0))
    cv2.putText(a, "v3", (20, 60), 0, 1.8, (0, 165, 255), 4); cv2.putText(b, "v4", (20, 60), 0, 1.8, (0, 255, 0), 4)
    side = cv2.resize(np.hstack([a, b]), (1600, 450))
    cv2.imwrite(f"{OUT}/cmp_{len(used)}.jpg", side, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(len(used), k, f, "v3:", [x["class_name"] for x in v3.get(f, [])], "v4:", [x["class_name"] for x in v4[f]])
