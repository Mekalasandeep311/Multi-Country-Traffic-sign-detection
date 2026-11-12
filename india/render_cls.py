"""Rendered rule-book signs for a label-checking classifier -> cls_data/{train,val}/<c|m><id>/.
c<id> = the sign as in the rule book; m<id> = its mirror image (only for signs that change when mirrored).
Rendered on random crops of real sign-free Indian road photos with perspective, rotation, scale, colour,
blur, low resolution and JPEG noise."""
import json, random, sys
from pathlib import Path
import cv2, numpy as np
R = Path(__file__).parent
C = json.load(open(R / "class_map.json"))["classes"]
rows = [json.loads(l) for l in open(R / "inventory.jsonl")]
bgs = [r["path"] for r in rows if r["kind"] == "neg"] + [r["path"] for r in rows if r["kind"] == "real" and not r["boxes"]]
rnd = random.Random(0)
S = 128

def rgba(p):
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im.ndim == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4:
        return im[:, :, :3], im[:, :, 3]
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)          # white page around the sign -> transparent (flood from border)
    m = np.zeros((g.shape[0] + 2, g.shape[1] + 2), np.uint8)
    wh = ((im.min(2) > 235) * 255).astype(np.uint8)
    for x, y in ((0, 0), (g.shape[1] - 1, 0), (0, g.shape[0] - 1), (g.shape[1] - 1, g.shape[0] - 1)):
        if wh[y, x]: cv2.floodFill(wh, m, (x, y), 128)
    a = np.where(wh == 128, 0, 255).astype(np.uint8)
    return im, a

T = []
for i in range(98):
    d = next((R / "templates").glob(f"{i:03d}_*"))
    T.append(rgba(next(d.glob("handbook.*"))))
asym = []
ASYM = float(sys.argv[2]) if len(sys.argv) > 2 else 18.0
score = []
for i, (im, a) in enumerate(T):
    ys, xs = np.where(a > 128)                       # crop to the sign's own outline so off-centring is ignored
    g = cv2.cvtColor(im[ys.min():ys.max() + 1, xs.min():xs.max() + 1], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (64, 64)).astype(np.float32)
    best = min(np.abs(g[:, max(0, d):64 + min(0, d)] - g[:, ::-1][:, max(0, -d):64 + min(0, -d)]).mean() for d in range(-3, 4))
    score.append(round(float(best), 1))

# signs whose mirror image is not another rule-book class (left/right pairs already cover each other)
asym = [4, 6, 10, 19, 51, 54, 82, 84, 88]
json.dump({"asym": asym, "score": score}, open(R / "asym.json", "w"))
if len(sys.argv) > 3: sys.exit(0)
print("asymmetric (mirror changes meaning)", len(asym))

bgcache = []
def bg():
    if len(bgcache) < 400 or rnd.random() < 0.02:
        im = cv2.imread(rnd.choice(bgs))
        if im is not None:
            H, W = im.shape[:2]; s = rnd.randint(min(H, W) // 8, min(H, W) // 2)
            x, y = rnd.randint(0, W - s), rnd.randint(0, H - s)
            bgcache.append(cv2.resize(im[y:y + s, x:x + s], (S, S)))
            if len(bgcache) > 600: bgcache.pop(0)
    return rnd.choice(bgcache).copy()

def render(im, a):
    out = bg()
    k = rnd.uniform(0.55, 0.92) * S
    h, w = im.shape[:2]; sc = k / max(h, w)
    sw, sh = max(4, int(w * sc * rnd.uniform(0.85, 1.15))), max(4, int(h * sc))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    j = lambda v: v + rnd.uniform(-0.08, 0.08) * k
    ox, oy = (S - sw) / 2 + rnd.uniform(-0.08, 0.08) * S, (S - sh) / 2 + rnd.uniform(-0.08, 0.08) * S
    dst = np.float32([[j(ox), j(oy)], [j(ox + sw), j(oy)], [j(ox + sw), j(oy + sh)], [j(ox), j(oy + sh)]])
    M = cv2.getPerspectiveTransform(src, dst)
    rot = cv2.getRotationMatrix2D((S / 2, S / 2), rnd.uniform(-10, 10), 1.0)
    M = np.vstack([rot, [0, 0, 1]]) @ M
    wi = cv2.warpPerspective(im, M, (S, S)); wa = cv2.warpPerspective(a, M, (S, S)).astype(np.float32)[..., None] / 255
    sign = wi.astype(np.float32)
    sign = sign * rnd.uniform(0.45, 1.15) + rnd.uniform(-35, 25)                  # light / shade
    sign = sign * np.float32([rnd.uniform(0.85, 1.15) for _ in range(3)])        # colour cast
    out = (out * (1 - wa) + np.clip(sign, 0, 255) * wa).astype(np.uint8)
    if rnd.random() < 0.6: out = cv2.GaussianBlur(out, (0, 0), rnd.uniform(0.3, 1.8))
    if rnd.random() < 0.5:
        r = rnd.randint(16, 64); out = cv2.resize(cv2.resize(out, (r, r), interpolation=cv2.INTER_AREA), (S, S))
    if rnd.random() < 0.7:
        _, b = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, rnd.randint(25, 90)]); out = cv2.imdecode(b, 1)
    return out

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
for i in range(98):
    vars_ = [("c", T[i])] + ([("m", (T[i][0][:, ::-1].copy(), T[i][1][:, ::-1].copy()))] if i in asym else [])
    for tag, (im, a) in vars_:
        for split, n in (("train", N), ("val", N // 8)):
            d = R / "cls_data" / split / f"{tag}{i:03d}"; d.mkdir(parents=True, exist_ok=True)
            for k in range(n):
                cv2.imwrite(str(d / f"{k:04d}.jpg"), render(im, a), [cv2.IMWRITE_JPEG_QUALITY, 92])
    if i % 10 == 0: print(i, flush=True)
print("done")
