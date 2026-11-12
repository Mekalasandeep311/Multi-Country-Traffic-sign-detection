"""Before/after sign-swap examples for Japan with the dataset's own make_synthetic()."""
import json, random, sys
from pathlib import Path
import cv2, numpy as np, yaml
ROOT = Path(".").resolve(); sys.path.insert(0, str(ROOT))
import build_balanced_dataset as bbd

OUT = ROOT / "swap_samples"; OUT.mkdir(exist_ok=True)
cmap = yaml.safe_load(open("class_map.yaml")); names, folders = cmap["names"], cmap["folders"]
cid_of = {v: k for k, v in names.items()}
PAIRS = [("No Parking", "Stop"), ("No U-turn", "Road Closed to Bicycles and Specified Motorized Bicycles"),
         ("Maximum Speed", "Maximum Weight"), ("Curve Ahead", "Deer Crossing"), ("Merge", "Two-Way Traffic"),
         ("Pedestrian Crossing", "Priority Road"), ("Prohibited from Proceeding except in Direction Indicated", "Bicycles Only (path)"),
         ("Fire Hydrant", "Slow")]


def draw(img, box, text, col):
    x0, y0, x1, y1 = map(int, box); img = img.copy()
    cv2.rectangle(img, (x0, y0), (x1, y1), col, 3)
    fs = 0.9; (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    ty = y0 - 8 if y0 - th - 14 > 0 else y1 + th + 12
    cv2.rectangle(img, (x0, ty - th - 6), (x0 + tw + 8, ty + 5), col, -1)
    cv2.putText(img, text, (x0 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def zoom(img, box, aspect=16 / 9):
    H, W = img.shape[:2]; x0, y0, x1, y1 = box
    h = max(360, 5 * (y1 - y0)); w = h * aspect
    a0 = int(np.clip((x0 + x1) / 2 - w / 2, 0, max(0, W - w))); b0 = int(np.clip((y0 + y1) / 2 - h / 2, 0, max(0, H - h)))
    return cv2.resize(img[b0:b0 + int(h), a0:a0 + int(w)], (800, 450))


rng = random.Random(4); sheet = []
for n, (src, dst) in enumerate(PAIRS, 1):
    res = json.loads((bbd.AUTO / f"{src}.json").read_text())
    cache = np.load(bbd.AUTO / "cache" / f"{src}.npz"); files = list(cache["files"])
    acc = [f for f, r in res.items() if r["status"] == "accepted" and 40 <= r["boxes"][0][3] - r["boxes"][0][1] <= 200]
    f = rng.choice(acc); b = res[f]["boxes"][0]; i = files.index(f)
    sel = (cache["img_idx"] == i) & (cache["sign"] >= bbd.OTHER_SIGN)
    donor = {"path": str(bbd.DATA / src / f), "box": b, "others": [x.tolist() for x in cache["boxes"][sel]], "cid": folders[src],
             "aspect": (b[2] - b[0]) / (b[3] - b[1])}
    dcid = cid_of[dst]
    tmpl = str(sorted(next((ROOT / "templates").glob(f"{dcid + 1:03d}_*")).glob("*.png"))[0])
    out = str(OUT / f"swap_{n}_after_raw.jpg")
    if bbd.make_synthetic((dcid, 0, "swap", donor, tmpl, out, 1000 + n)) is None:
        continue
    before = cv2.imread(donor["path"]); after = cv2.imread(out)
    H, W = after.shape[:2]; c, x, y, w, h = map(float, Path(out).with_suffix(".txt").read_text().split())
    ab = [(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]
    bz = zoom(draw(before, b, f"real: {src[:40]}", (0, 220, 0)), b)
    az = zoom(draw(after, ab, f"synthetic: {dst[:40]}", (0, 165, 255)), b)
    arrow = np.full((450, 60, 3), 255, np.uint8); cv2.arrowedLine(arrow, (8, 225), (52, 225), (40, 40, 200), 6, tipLength=0.5)
    row = np.hstack([bz, arrow, az]); cv2.imwrite(str(OUT / f"swap_{n}_{dst[:30].replace(' ', '_')}.jpg"), row)
    sheet.append(row); Path(out).unlink(); Path(out).with_suffix(".txt").unlink()
    print(n, src, "->", dst, f)
cv2.imwrite(str(OUT / "all_swaps.jpg"), cv2.resize(np.vstack(sheet), None, fx=0.6, fy=0.6))
