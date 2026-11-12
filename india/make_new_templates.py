"""Templates cut from the test-bed video: Charging station (100), Width limit (101), red-triangle Tunnel (extra for 75)."""
import json, cv2, numpy as np
from pathlib import Path
def best(v, i):
    t = {x["id"]: x for x in json.load(open(f"india_mine/{v}/tracks.json"))}[i]
    H, W = 1200, 1920
    ok = [(f, b) for f, b in t["boxes"] if b[0] > 4 and b[1] > 4 and b[2] < W - 4 and b[3] < H - 4]
    f, b = max(ok, key=lambda fb: (fb[1][2] - fb[1][0]) * (fb[1][3] - fb[1][1]))
    im = cv2.imread(f"india_mine/{v}/frames/f{f:06d}.jpg"); x0, y0, x1, y1 = map(int, b[:4])
    return im[y0:y1, x0:x1].copy()
def save(name, im, mask):
    d = Path("templates") / name; d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "handbook.png"), np.dstack([im, mask])); print(name, im.shape)
a = best("cam_20261106_100849", 2993); save("100_Charging station", a, np.full(a.shape[:2], 255, np.uint8))
b = best("cam_20261106_100001", 4378); m = np.zeros(b.shape[:2], np.uint8); h, w = m.shape
cv2.ellipse(m, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1); m[0, 0] = 0; save("101_Width limit", b, m)
c = best("cam_20261106_100001", 478); m = np.zeros(c.shape[:2], np.uint8); h, w = m.shape
cv2.fillPoly(m, [np.array([[w // 2, 0], [w - 1, h - 1], [0, h - 1]])], 255); d = Path("templates_extra") / "075_Tunnel triangle"; d.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(d / "handbook.png"), np.dstack([c, m])); print("tunnel triangle", c.shape)
tl = []
for im, name in ((a, "c"), (b, "w"), (c, "t")):
    tl.append(cv2.resize(im, (260, 260)))
cv2.imwrite("review/new_templates.jpg", np.hstack(tl))
