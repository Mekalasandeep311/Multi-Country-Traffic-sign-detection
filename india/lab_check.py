import glob, json, random, cv2, numpy as np, collections
C = json.load(open("class_map.json"))["classes"] + ["Charging station", "Width limit"]
if len(C) > 102: C = C[:100] + ["Charging station", "Width limit"]
items = []; cnt = collections.Counter()
for t in glob.glob("india_labeled/*/*.txt"):
    for l in open(t):
        p = l.split(); items.append((t, p)); cnt[int(p[0])] += 1
print("boxes per class", sorted(cnt.items()))
random.seed(11); tiles = []
for t, p in random.sample(items, 60):
    im = cv2.imread(t[:-4] + ".jpg"); H, W = im.shape[:2]; cx, cy, w, h = map(float, p[1:5])
    x0, y0, x1, y1 = int((cx-w/2)*W), int((cy-h/2)*H), int((cx+w/2)*W), int((cy+h/2)*H); m = int(max(x1-x0, y1-y0) * .5) + 6
    X0, Y0 = max(0, x0-m), max(0, y0-m); cr = im[Y0:y1+m, X0:x1+m].copy()
    cv2.rectangle(cr, (x0-X0, y0-Y0), (x1-X0, y1-Y0), (0, 255, 0), max(1, cr.shape[1]//100))
    c = cv2.resize(cr, (160, 160)); cv2.rectangle(c, (0, 0), (160, 16), (0, 0, 0), -1); cv2.putText(c, C[int(p[0])][:24], (2, 12), 0, 0.36, (255, 255, 255), 1); tiles.append(c)
cv2.imwrite("review/lab_check.jpg", np.vstack([np.hstack(tiles[i:i+10]) for i in range(0, 60, 10)]))
