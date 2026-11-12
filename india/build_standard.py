"""standard_india/<NNN>_<class>/ : 400 images per rule-book class (99) with YOLO labels.
Real photos first, then web pictures (train only), the rest sign-swap synthetic on real Indian road frames.
Also writes split.json: real-only val / test images (grouped by frame-number blocks so near frames stay together)."""
import json, random, re, sys, collections
from pathlib import Path
from multiprocessing import Pool
import cv2, numpy as np

R = Path(__file__).parent
CM = json.load(open(R / "class_map.json")); C = CM["classes"]
OUT = R / "standard_india"; N = 400; LONG = 1920
rows = [json.loads(l) for l in open(R / "clean_labels.jsonl")]
real = [r for r in rows if r["kind"] == "real"]
neg = [r for r in rows if r["kind"] == "neg"]


def cutout(p):
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4 and im[:, :, 3].min() < 250:
        rgb, a = im[:, :, :3].copy(), im[:, :, 3].copy()
    else:
        rgb = im[:, :, :3].copy(); h, w = rgb.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        seeds = ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        for x, y in seeds:                         # only white / grey / black page backgrounds, never a coloured panel
            if mask[y + 1, x + 1] == 0 and hsv[y, x, 1] < 40:
                cv2.floodFill(rgb.copy(), mask, (x, y), 0, (18, 18, 18), (18, 18, 18), cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8))
        a = np.where(mask[1:-1, 1:-1] > 0, 0, 255).astype(np.uint8)
        if (a > 0).mean() < 0.2:          # flood ate the sign: keep the whole picture
            a[:] = 255
    ys, xs = np.where(a > 0)
    rgb = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return rgb, cv2.GaussianBlur(a, (3, 3), 0)


def plate(rgb, a):
    """red DO NOT LITTER plate under the disc; returns canvas, alpha, disc height"""
    h, w = rgb.shape[:2]; ph = int(h * 0.28); gap = int(h * 0.06)
    canvas = np.zeros((h + gap + ph, w, 3), np.uint8); alpha = np.zeros((h + gap + ph, w), np.uint8)
    canvas[:h], alpha[:h] = rgb, a
    x0, x1 = int(w * 0.08), int(w * 0.92)
    canvas[h + gap:, x0:x1] = (40, 30, 200); alpha[h + gap:, x0:x1] = 255
    cv2.rectangle(canvas, (x0 + 3, h + gap + 3), (x1 - 4, h + gap + ph - 4), (255, 255, 255), max(1, ph // 14))
    txt = "DO NOT LITTER"; fs = ph / 40.0; th_ = max(1, int(fs * 2))
    (tw, tth), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, fs, th_)
    fs *= min(1.0, (x1 - x0) * 0.88 / tw); th_ = max(1, int(fs * 2))
    (tw, tth), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, fs, th_)
    cv2.putText(canvas, txt, ((w - tw) // 2, h + gap + (ph + tth) // 2), cv2.FONT_HERSHEY_DUPLEX, fs, (255, 255, 255), th_, cv2.LINE_AA)
    return canvas, alpha, h


T = {}
for i in range(99):
    d = next((R / "templates").glob(f"{i:03d}_*"))
    T[i] = cutout(next(d.glob("handbook.*")))


def load_scaled(path):
    im = cv2.imread(path); s = LONG / max(im.shape[:2])
    if s < 1:
        im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return im


def paste(img, c, box, rnd):
    """paste class c into box (x0, y0, x1, y1); returns the label box or None"""
    rgb, a = T[c]; disc_h = None
    if c == 98:
        rgb, a, disc_h = plate(rgb, a)
    x0, y0, x1, y1 = box; bw, bh = x1 - x0, y1 - y0
    th, tw = rgb.shape[:2]; s = min(bw / tw, bh / th) * rnd.uniform(0.9, 1.05)
    nw, nh = max(6, int(tw * s)), max(6, int(th * s))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    px, py = int(cx - nw / 2), int(cy - nh / 2)
    H, W = img.shape[:2]
    if px < 0 or py < 0 or px + nw > W or py + nh > H:
        return None
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    def j():
        return rnd.uniform(-0.06, 0.06) * min(nw, nh)
    dst = np.float32([[j(), j()], [nw + j(), j()], [nw + j(), nh + j()], [j(), nh + j()]])
    M = cv2.getPerspectiveTransform(src, dst)
    sg = cv2.warpPerspective(rgb, M, (nw, nh), flags=cv2.INTER_AREA).astype(np.float32)
    sa = cv2.warpPerspective(a, M, (nw, nh)).astype(np.float32)[..., None] / 255
    reg = img[py:py + nh, px:px + nw].astype(np.float32)
    lum = reg.mean() / 128.0                       # match scene light, blur and noise
    sg = sg * np.clip(lum * rnd.uniform(0.75, 1.1), 0.35, 1.25) + rnd.uniform(-15, 10)
    k = max(0.0, rnd.gauss(0.6, 0.5) * (nw / 60.0) ** 0.5)
    if k > 0.2:
        sg = cv2.GaussianBlur(sg, (0, 0), k)
    sg += np.random.default_rng(rnd.randint(0, 1 << 30)).normal(0, rnd.uniform(1, 6), sg.shape)
    img[py:py + nh, px:px + nw] = np.clip(reg * (1 - sa) + sg * sa, 0, 255).astype(np.uint8)
    lab_h = nh if disc_h is None else int(nh * disc_h / th)
    return (px, py, px + nw, py + lab_h)


def erase(img, boxes):
    m = np.zeros(img.shape[:2], np.uint8)
    for (x0, y0, x1, y1) in boxes:
        pw, ph = int((x1 - x0) * 0.12) + 2, int((y1 - y0) * 0.12) + 2
        m[max(0, y0 - ph):y1 + ph, max(0, x0 - pw):x1 + pw] = 255
    return cv2.inpaint(img, m, 5, cv2.INPAINT_TELEA)


def yolo(c, b, W, H):
    x0, y0, x1, y1 = b
    return f"{c} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n"


def target_box(c, ccx, ccy, side):
    th, tw = T[c][0].shape[:2]; asp = tw / th
    bw, bh = (side, side / asp) if asp >= 1 else (side * asp, side)
    if c == 98:
        bh *= 1.4
    return (int(ccx - bw / 2), int(ccy - bh / 2), int(ccx + bw / 2), int(ccy + bh / 2))


def synth(c, rnd):
    for _ in range(20):
        if rnd.random() < 0.7:                     # swap into a real frame where a sign stood
            r = rnd.choice(real); img = load_scaled(r["path"]); H, W = img.shape[:2]
            bx = [(int((cx - w / 2) * W), int((cy - h / 2) * H), int((cx + w / 2) * W), int((cy + h / 2) * H))
                  for _, cx, cy, w, h in r["boxes"]]
            big = [b for b in bx if b[2] - b[0] >= 16]
            if not big:
                continue
            img = erase(img, bx); tgt = rnd.choice(big)
            side = max(tgt[2] - tgt[0], tgt[3] - tgt[1]) * rnd.uniform(0.9, 1.3)
            box = target_box(c, (tgt[0] + tgt[2]) / 2, (tgt[1] + tgt[3]) / 2, side)
        else:                                      # sign on a pole in a sign-free road frame
            r = rnd.choice(neg); img = load_scaled(r["path"]); H, W = img.shape[:2]
            side = rnd.uniform(0.02, 0.09) * W
            ccx = rnd.choice([rnd.uniform(0.05, 0.3), rnd.uniform(0.7, 0.95)]) * W
            ccy = rnd.uniform(0.25, 0.55) * H
            box = target_box(c, ccx, ccy, side)
            g = rnd.randint(90, 160)
            cv2.line(img, (int(ccx), box[3]), (int(ccx), min(H - 1, int(box[3] + side * rnd.uniform(1.5, 3)))), (g, g, g), max(2, int(side / 14)))
        lab = paste(img, c, box, rnd)
        if lab:
            return img, [yolo(c, lab, img.shape[1], img.shape[0])]
    return None, None


def block(r):
    s = Path(r["path"]).stem; m = re.search(r"(\d+)", s)
    return (s[:3], int(m.group(1)) // 40) if m else (s, 0)


def make_split():
    rnd = random.Random(7); blocks = collections.defaultdict(list)
    for r in real:
        blocks[block(r)].append(r)
    keys = sorted(blocks); rnd.shuffle(keys)
    need = collections.Counter(c for r in real for c in {b[0] for b in r["boxes"]})
    got_v, got_t = collections.Counter(), collections.Counter(); val, test = set(), set()
    for kk in keys:
        cs = {b[0] for r in blocks[kk] for b in r["boxes"] if b[0] < 99}
        if not cs or not all(need[c] >= 5 for c in cs):
            continue
        for got, dest in ((got_v, val), (got_t, test)):
            if any(got[c] < max(1, 0.1 * need[c]) for c in cs):
                dest |= {r["path"] for r in blocks[kk]}
                for r in blocks[kk]:
                    got.update({b[0] for b in r["boxes"]})
                break
    return val, test


SPLIT = R / "split.json"
if SPLIT.exists():
    _s = json.load(open(SPLIT)); HOLD = set(_s["val"]) | set(_s["test"])
else:
    HOLD = set()


def build(c):
    rnd = random.Random(1000 + c)
    d = OUT / f"{c:03d}_{C[c].replace('/', '-')}"; d.mkdir(parents=True, exist_ok=True)
    own = [r for r in rows if r["kind"] in ("real", "web") and any(b[0] == c for b in r["boxes"]) and r["path"] not in HOLD]
    rnd.shuffle(own); own.sort(key=lambda r: r["kind"] != "real")
    n_real = n_web = n_syn = 0
    for r in own[:N]:
        img = load_scaled(r["path"]); s = Path(r["path"]).stem
        bs = sorted(r["boxes"], key=lambda b: b[0] != c)
        cv2.imwrite(str(d / f"{r['kind']}_{s}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (d / f"{r['kind']}_{s}.txt").write_text("".join(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n" for b in bs))
        n_real += r["kind"] == "real"; n_web += r["kind"] == "web"
    tries = 0
    while n_real + n_web + n_syn < N and tries < 4 * N:
        img, lab = synth(c, rnd); tries += 1
        if img is None:
            continue
        cv2.imwrite(str(d / f"syn_{n_syn:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (d / f"syn_{n_syn:05d}.txt").write_text("".join(lab)); n_syn += 1
    return c, n_real, n_web, n_syn


if __name__ == "__main__":
    if sys.argv[1] == "test":                      # a few synthetic samples for checking
        rnd = random.Random(0)
        for c in map(int, sys.argv[2].split(",")):
            img, lab = synth(c, rnd)
            x = lab[0].split(); H, W = img.shape[:2]
            cx, cy, w, h = (float(v) for v in x[1:5])
            cv2.rectangle(img, (int((cx - w / 2) * W), int((cy - h / 2) * H)), (int((cx + w / 2) * W), int((cy + h / 2) * H)), (0, 255, 0), 2)
            m = int(max(w * W, h * H) * 2.5)
            crop = img[max(0, int(cy * H) - m):int(cy * H) + m, max(0, int(cx * W) - m):int(cx * W) + m]
            cv2.imwrite(str(R / "review" / f"syn_test_{c}.jpg"), cv2.resize(crop, (300, 300)))
        sys.exit(0)
    val, test = make_split()
    json.dump({"val": sorted(val), "test": sorted(test)}, open(SPLIT, "w"))
    HOLD = val | test
    print("val images", len(val), "test images", len(test), flush=True)
    with Pool(int(sys.argv[1])) as p:
        res = p.map(build, range(99), chunksize=1)
    with open(OUT / "report.csv", "w") as fh:
        fh.write("id,class,real,web,synthetic\n")
        for c, a, b, s in res:
            fh.write(f"{c},{C[c]},{a},{b},{s}\n")
    print("done real", sum(r[1] for r in res), "web", sum(r[2] for r in res), "synthetic", sum(r[3] for r in res))
