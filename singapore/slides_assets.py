"""Collect material for the project slides (up to the v2 model) into slides_assets/.

  swap_<n>_before.jpg / _after.jpg   real frame with its auto-boxed sign, and the synthetic version
                                     where that sign was erased and a missing class's handbook sign
                                     pasted in its place (same make_synthetic() used for the dataset)
  autobox_examples.jpg               auto-boxed real signs (folder sign only)
  templates.jpg                      handbook templates
  per_class.csv                      v2 per-class P/R/AP50/AP50-95 on the validation set
  stats.json                         dataset numbers
  results.png, confusion_matrix_normalized.png, detections_*.jpg
"""
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import build_balanced_dataset as bbd  # noqa: E402

OUT = ROOT / "slides_assets"
OUT.mkdir(exist_ok=True)
DATA = ROOT / "Traffic Signs Singapore"
AUTO = ROOT / "autobox"
BAL = ROOT / "sg_signs_balanced"


def draw(img, box, text, col=(0, 220, 0)):
    x0, y0, x1, y1 = map(int, box)
    img = img.copy()
    t = max(2, img.shape[1] // 500)
    cv2.rectangle(img, (x0, y0), (x1, y1), col, t)
    fs = img.shape[1] / 1600
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    ty = y0 - 6 if y0 - th - 12 > 0 else y1 + th + 10
    cv2.rectangle(img, (x0, ty - th - 6), (x0 + tw + 8, ty + 4), col, -1)
    cv2.putText(img, text, (x0 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def crop_around(img, box, scale=4.0, aspect=16 / 9):
    """Zoomed view around a box so a small sign is visible on a slide."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    h = max(220, scale * (y1 - y0)); w = h * aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a0, b0 = int(np.clip(cx - w / 2, 0, max(0, W - w))), int(np.clip(cy - h / 2, 0, max(0, H - h)))
    return img[b0:b0 + int(h), a0:a0 + int(w)], (a0, b0)


def sign_swaps(names, folders):
    """Generate before/after pairs with the dataset's own make_synthetic()."""
    pairs = [("Keep left", "Turn left only"), ("No entry for all vehicles", "No sounding of horn"),
             ("Pedestrian crossing prohibition", "Speed limit 90 km"), ("Split traffic", "Roundabout"),
             ("Stop", "Give Way to pedestrians"), ("Curve Alignment Markers", "Slippery road")]
    cid_of = {v: k for k, v in names.items()}
    rng = random.Random(3)
    n = 0
    for donor_cls, target_cls in pairs:
        res = json.loads((AUTO / f"{donor_cls}.json").read_text())
        cache = np.load(AUTO / "cache" / f"{donor_cls}.npz")
        files = list(cache["files"])
        cands = [(f, r) for f, r in res.items() if r["status"] == "accepted"
                 and 45 <= r["boxes"][0][3] - r["boxes"][0][1] <= 160]
        if not cands:
            continue
        f, r = rng.choice(cands)
        i = files.index(f)
        sel = (cache["img_idx"] == i) & (cache["sign"] >= bbd.OTHER_SIGN)
        donor = {"path": str(DATA / donor_cls / f), "box": r["boxes"][0],
                 "others": [b.tolist() for b in cache["boxes"][sel]] + r["boxes"][1:]}
        tcid = cid_of[target_cls]
        tmpl = [str(p) for d in (ROOT / "templates").glob(f"{tcid + 1:03d}_*") for p in d.glob("*.png")][0]
        n += 1
        after_p = OUT / f"swap_{n}_after.jpg"
        if bbd.make_synthetic((tcid, 0, "swap", donor, tmpl, str(after_p), 1000 + n)) is None:
            continue
        before = cv2.imread(donor["path"])
        after = cv2.imread(str(after_p))
        H, W = after.shape[:2]
        c, x, y, w, h = map(float, after_p.with_suffix(".txt").read_text().split())
        abox = [(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]
        zb, (ox, oy) = crop_around(before, donor["box"])
        za = after[oy:oy + zb.shape[0], ox:ox + zb.shape[1]]
        sh = lambda b: [b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy]
        cv2.imwrite(str(OUT / f"swap_{n}_before.jpg"), draw(zb, sh(donor["box"]), f"real: {donor_cls}"))
        cv2.imwrite(str(after_p), draw(za, sh(abox), f"synthetic: {target_cls}", (0, 165, 255)))
        cv2.imwrite(str(OUT / f"swap_{n}_after_full.jpg"), after)
        after_p.with_suffix(".txt").unlink()
        cv2.imwrite(str(OUT / f"swap_{n}_template.png"), cv2.imread(tmpl, cv2.IMREAD_UNCHANGED))
        print("swap", n, donor_cls, "->", target_cls)


def autobox_examples(folders):
    tiles = []
    rng = random.Random(5)
    for folder in ["Keep left", "No entry for all vehicles", "Pedestrian crossing prohibition", "Stop",
                   "Curve Alignment Markers", "Speed limit 40 km", "Advance Directional Sign (non-expressway)",
                   "Works Ahead", "Height limit (4.5m)", "Slow", "No right turn", "Give Way"]:
        res = json.loads((AUTO / f"{folder}.json").read_text())
        acc = [(f, r) for f, r in res.items() if r["status"] == "accepted" and r["boxes"][0][3] - r["boxes"][0][1] > 30]
        f, r = rng.choice(acc)
        im = cv2.imread(str(DATA / folder / f))
        z, (ox, oy) = crop_around(im, r["boxes"][0], scale=5)
        b = r["boxes"][0]
        z = draw(z, [b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy], folder.split(" (")[0][:26])
        tiles.append(cv2.resize(z, (480, 270)))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 12, 4)])
    cv2.imwrite(str(OUT / "autobox_examples.jpg"), grid)


def templates_grid():
    ps = sorted((ROOT / "templates").glob("*/0.png"))
    pick = [ps[i] for i in np.linspace(0, len(ps) - 1, 24).astype(int)]
    tiles = []
    for p in pick:
        rgba = bbd.load_template_rgba(p)
        t = rgba[..., :3].copy(); a = rgba[..., 3:4] / 255
        t = (t * a + 255 * (1 - a)).astype(np.uint8)
        h, w = t.shape[:2]; s = 150 / max(h, w)
        t = cv2.resize(t, (int(w * s), int(h * s)))
        c = np.full((170, 170, 3), 255, np.uint8)
        y, x = (170 - t.shape[0]) // 2, (170 - t.shape[1]) // 2
        c[y:y + t.shape[0], x:x + t.shape[1]] = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
        tiles.append(c)
    cv2.imwrite(str(OUT / "templates.jpg"), np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, 24, 8)]))


def per_class():
    from ultralytics import YOLO
    m = YOLO(str(ROOT / "runs" / "sg_signs_v2" / "weights" / "best.pt"))
    r = m.val(data=str(ROOT / "sg_signs_train" / "data.yaml"), imgsz=640, batch=32, plots=False, verbose=False,
              project=str(OUT / "val"), name="v2", exist_ok=True)
    real_used = {}
    with open(BAL / "report.csv") as fh:
        for row in csv.DictReader(fh):
            real_used[row["class"]] = int(row["real_used"])
    with open(OUT / "per_class.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "real_images", "instances", "P", "R", "AP50", "AP50_95"])
        nt = r.nt_per_class if hasattr(r, "nt_per_class") else None
        for j, c in enumerate(r.box.ap_class_index):
            p, rc, ap50, ap = r.box.class_result(j)
            name = m.names[int(c)]
            w.writerow([name, real_used.get(name, ""), int(nt[c]) if nt is not None else "",
                        round(p, 3), round(rc, 3), round(ap50, 3), round(ap, 3)])
    return {"P": r.box.mp, "R": r.box.mr, "mAP50": r.box.map50, "mAP50_95": r.box.map}


def main():
    cm = yaml.safe_load((ROOT / "class_map.yaml").read_text())
    names, folders = cm["names"], cm["folders"]
    sign_swaps(names, folders)
    autobox_examples(folders)
    templates_grid()
    for f in ("results.png", "confusion_matrix_normalized.png", "args.yaml", "results.csv"):
        if (ROOT / "runs" / "sg_signs_v2" / f).exists():
            shutil.copy(ROOT / "runs" / "sg_signs_v2" / f, OUT / f"v2_{f}")
    for i, f in enumerate(sorted((ROOT / "video" / "video1_last30min_detected_frames").glob("*.jpg"))[::4000][:8]):
        shutil.copy(f, OUT / f"detections_{i}.jpg")
    overall = per_class()

    rows = list(csv.DictReader(open(BAL / "report.csv")))
    stats = {
        "classes": len(names), "handbook_classes": 156, "extra_classes": len(names) - 156,
        "real_images_total": sum(1 for _ in DATA.glob("*/*.jpg")),
        "real_folders": len(folders),
        "real_labeled": sum(1 for _ in DATA.glob("*/*.txt")),
        "per_class_target": 400,
        "balanced_total": sum(1 for _ in BAL.glob("*/*.jpg")),
        "real_used": sum(int(r["real_used"]) for r in rows),
        "synthetic": sum(int(r["synthetic"]) for r in rows),
        "real_extra": sum(int(r["real_extra"]) for r in rows),
        "boston_excluded": sum(int(r["boston_excluded"]) for r in rows),
        "fully_synthetic_classes": sum(int(r["real_used"]) == 0 for r in rows),
        "v2_val": overall,
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
