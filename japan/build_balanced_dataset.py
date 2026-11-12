"""Build the balanced per-class dataset (real auto-boxed + single-sign synthetic).

standard_japan/
  classes.txt
  <Class name>/real_<orig>.jpg + .txt        confident real labels, up to TARGET
  <Class name>/syn_<n>.jpg + .txt            synthetic, exactly one sign, fills the class to TARGET
  <Class name>/extra/real_<orig>.jpg + .txt  surplus confident real images (never deleted)
  _unlabeled/<Class name>/...                uncertain / rejected real frames (round 2)
  _review/<Class name>_synthetic.jpg         sample of synthetic images per class
  report.csv

Synthetic = "sign swap" (SWAP_SHARE): a real Japanese frame whose sign was auto-boxed; that sign and
any other sign-like boxes are inpainted, and the missing class's sign is put into the donor sign's
place (similar aspect ratio), matched to the donor's brightness, sharpness and noise.
With SWAP_SHARE < 1 the rest would be a sign-free real frame + one sign on a pole (off for Japan).
Original dataset folders are only read.

usage: python build_balanced_dataset.py [target]
"""
import csv
import json
import os
import random
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml

from autolabel import load_template_rgba, warp_sign

ROOT = Path(__file__).parent
DATA = ROOT / "classified Data japan" / "Merged"
AUTO = ROOT / "autobox"
OUT = Path(os.environ.get("BALANCED_OUT", ROOT / "standard_japan"))   # override to build elsewhere
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 400
# Hard-mining: SYN_ONLY="cid:n,cid:n" builds ONLY n extra synthetic images for those classes (no real
# images, no other classes) into BALANCED_OUT, e.g. for classes the model confuses.
SYN_ONLY = {int(a): int(b) for a, b in (x.split(":") for x in os.environ.get("SYN_ONLY", "").split(",") if x)}
SWAP_SHARE = 1.0          # Japan: every synthetic image is a sign swap in a real frame (user request)
MAX_DONOR_H = 220          # donor signs taller than this leave big inpainting smears
OTHER_SIGN = 0.20          # OWLv2 sign-ness above which a box in a donor frame is erased too
HASH_DIST = 10             # dHash distance that counts as "different enough" when picking real frames


def safe(name):
    return name.replace("/", "-").replace(":", "-").strip()


def dhash(path):
    im = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    s = cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA)
    return int(np.packbits((s[:, 1:] > s[:, :-1]).flatten()).view(">u8")[0])


def pick_diverse(files, k):
    """Up to k files, preferring frames that differ from everything already picked."""
    if len(files) <= k:
        return list(files), []
    hashes = {f: dhash(f) for f in files}
    chosen = []
    for f in files:
        if all(bin(hashes[f] ^ hashes[c]).count("1") >= HASH_DIST for c in chosen):
            chosen.append(f)
            if len(chosen) == k:
                break
    rest = [f for f in files if f not in set(chosen)]
    if len(chosen) < k:
        fill = [rest[i] for i in np.linspace(0, len(rest) - 1, k - len(chosen)).astype(int)]
        chosen += fill
        rest = [f for f in rest if f not in set(fill)]
    return chosen, rest


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# ---------------------------------------------------------------- synthetic helpers
def inpaint_boxes(img, boxes, grow=0.15):
    """Inpaint each box (+margin) inside a local ROI, which is much faster than full-frame inpainting."""
    H, W = img.shape[:2]
    for x0, y0, x1, y1 in boxes:
        m = grow * max(x1 - x0, y1 - y0) + 2
        bx0, by0, bx1, by1 = int(max(0, x0 - m)), int(max(0, y0 - m)), int(min(W, x1 + m)), int(min(H, y1 + m))
        r = int(max(bx1 - bx0, by1 - by0))
        rx0, ry0, rx1, ry1 = max(0, bx0 - r), max(0, by0 - r), min(W, bx1 + r), min(H, by1 + r)
        roi = img[ry0:ry1, rx0:rx1]
        mask = np.zeros(roi.shape[:2], np.uint8)
        mask[by0 - ry0:by1 - ry0, bx0 - rx0:bx1 - rx0] = 255
        img[ry0:ry1, rx0:rx1] = cv2.inpaint(roi, mask, 5, cv2.INPAINT_TELEA)
    return img


def sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var() if gray.size else 0.0


def match_appearance(sign_rgba, donor_crop, rng):
    """Luminance mean/std transfer (colour hue kept), blur + noise matched to the donor sign."""
    rgb = sign_rgba[..., :3]
    a = sign_rgba[..., 3] > 127
    if donor_crop.size and a.any():
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        dl = cv2.cvtColor(donor_crop, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
        L = lab[..., 0]
        mu, sd = L[a].mean(), L[a].std() + 1e-3
        target = (L - mu) / sd * (0.5 * sd + 0.5 * (dl.std() + 1e-3)) + (0.4 * mu + 0.6 * dl.mean())
        lab[..., 0] = np.clip(target, 0, 255)
        rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        # blur until the pasted sign is about as sharp as the donor sign
        want = sharpness(cv2.cvtColor(donor_crop, cv2.COLOR_RGB2GRAY))
        best, best_d = rgb, abs(sharpness(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)) - want)
        for s in (0.5, 0.8, 1.1, 1.5, 2.0):
            b = cv2.GaussianBlur(rgb, (0, 0), s)
            d = abs(sharpness(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)) - want)
            if d < best_d:
                best, best_d = b, d
        rgb = best
        noise = np.std(donor_crop.astype(np.float32) - cv2.GaussianBlur(donor_crop, (0, 0), 1.0).astype(np.float32))
        rgb = np.clip(rgb.astype(np.float32) + rng.normal(0, min(noise, 12), rgb.shape), 0, 255).astype(np.uint8)
    return np.dstack([rgb, sign_rgba[..., 3]])


def paste(img, sign, cx, cy):
    """Alpha-paste RGBA `sign` centred at (cx, cy); returns tight box of the pasted pixels or None."""
    H, W = img.shape[:2]
    sh, sw = sign.shape[:2]
    x0, y0 = int(round(cx - sw / 2)), int(round(cy - sh / 2))
    ix0, iy0, ix1, iy1 = max(0, x0), max(0, y0), min(W, x0 + sw), min(H, y0 + sh)
    if ix1 - ix0 < 4 or iy1 - iy0 < 4:
        return None
    s = sign[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
    a = s[..., 3:4].astype(np.float32) / 255
    img[iy0:iy1, ix0:ix1] = (img[iy0:iy1, ix0:ix1] * (1 - a) + s[..., :3] * a).astype(np.uint8)
    ys, xs = np.nonzero(s[..., 3] > 127)
    if not len(xs):
        return None
    return ix0 + xs.min(), iy0 + ys.min(), ix0 + xs.max() + 1, iy0 + ys.max() + 1


def draw_pole(img, cx, top, sign_h, rng):
    H = img.shape[0]
    w = max(2, int(0.07 * sign_h))
    bottom = int(min(H, top + sign_h * rng.uniform(2.0, 4.0)))
    shade = int(rng.integers(90, 160))
    cv2.rectangle(img, (int(cx - w / 2), int(top)), (int(cx + w / 2), bottom), (shade, shade, shade), -1)


def make_synthetic(task):
    (cid, k, mode, donor, template, out_img, seed) = task
    rng = np.random.default_rng(seed)
    img = cv2.cvtColor(cv2.imread(donor["path"]), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    if "_templates_real" in template:      # crop of a real sign: opaque rectangle
        t = cv2.cvtColor(cv2.imread(template), cv2.COLOR_BGR2RGB)
        tmpl = np.dstack([t, np.full(t.shape[:2], 255, np.uint8)])
    else:                                  # handbook graphic: white page background made transparent
        tmpl = load_template_rgba(template)
    x0, y0, x1, y1 = donor["box"]
    donor_crop = img[int(y0):int(y1), int(x0):int(x1)].copy()
    img = inpaint_boxes(img, [donor["box"]] + donor["others"])

    if mode == "swap":
        bh = y1 - y0
        sign = warp_sign(tmpl, max(10, int(round(bh))), rng)
        sign = match_appearance(sign, donor_crop, rng)
        box = paste(img, sign, (x0 + x1) / 2, (y0 + y1) / 2)
    else:
        h = int(np.exp(rng.uniform(np.log(18), np.log(160))))
        sign = warp_sign(tmpl, h, rng)
        sign = match_appearance(sign, donor_crop, rng)
        sh, sw = sign.shape[:2]
        cx = rng.uniform(0.05 * W + sw / 2, 0.95 * W - sw / 2) if 0.9 * W > sw else W / 2
        cy = rng.uniform(0.25 * H, 0.6 * H)
        draw_pole(img, cx, cy + sh / 2 - 2, sh, rng)
        box = paste(img, sign, cx, cy)
    if box is None:
        return None
    bx0, by0, bx1, by1 = box
    cv2.imwrite(out_img, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(85, 96))])
    Path(out_img).with_suffix(".txt").write_text(
        f"{cid} {(bx0 + bx1) / 2 / W:.6f} {(by0 + by1) / 2 / H:.6f} {(bx1 - bx0) / W:.6f} {(by1 - by0) / H:.6f}\n")
    return out_img


# ---------------------------------------------------------------- main
def main():
    cmap = yaml.safe_load((ROOT / "class_map.yaml").read_text())
    names, folders = cmap["names"], cmap["folders"]
    boston = set()
    if (ROOT / "nusc_city.csv").exists():
        with open(ROOT / "nusc_city.csv") as fh:
            boston = {(r["class"], r["file"]) for r in csv.DictReader(fh) if r["city"] == "boston"}

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    (OUT / "classes.txt").write_text("\n".join(names[i] for i in range(len(names))) + "\n")

    real = {cid: [] for cid in names}      # cid -> [Path of accepted SG image]
    donors = []                            # accepted SG signs usable as swap donors
    report = {cid: {"real_found": 0, "boston": 0, "unlabeled": 0} for cid in names}
    for folder, cid in folders.items():
        res_file = AUTO / f"{folder}.json"
        if not res_file.exists():
            print("no autobox result for", folder); continue
        res = json.loads(res_file.read_text())
        cache = np.load(AUTO / "cache" / f"{folder}.npz")
        files = list(cache["files"])
        cname = safe(names[cid])
        for f, r in res.items():
            src = DATA / folder / f
            if (folder, f) in boston:
                report[cid]["boston"] += 1
                link(src, OUT / "_boston" / cname / f)
                if (src.with_suffix(".txt")).exists():
                    link(src.with_suffix(".txt"), OUT / "_boston" / cname / (Path(f).stem + ".txt"))
                continue
            if r["status"] != "accepted":
                report[cid]["unlabeled"] += 1
                link(src, OUT / "_unlabeled" / cname / f)
                continue
            report[cid]["real_found"] += 1
            real[cid].append(src)
            b = r["boxes"][0]
            if 12 <= b[3] - b[1] <= MAX_DONOR_H:
                i = files.index(f)
                sel = (cache["img_idx"] == i) & (cache["sign"] >= OTHER_SIGN)
                others = [bb.tolist() for bb in cache["boxes"][sel]] + [bb for bb in r["boxes"][1:]]
                donors.append({"path": str(src), "box": b, "others": others, "cid": cid,
                               "aspect": (b[2] - b[0]) / max(1e-3, b[3] - b[1])})
    print(f"donor signs: {len(donors)}", flush=True)

    # templates per class: handbook graphics, else crops of confident real boxes
    tmpl_dir = OUT / "_templates_real"
    templates = {}
    for cid in names:
        pngs = [str(p) for d in (ROOT / "templates").glob(f"{cid + 1:03d}_*") for p in d.glob("*.png")]
        if not pngs and real[cid]:
            folder = next(f for f, c in folders.items() if c == cid)
            res = json.loads((AUTO / f"{folder}.json").read_text())
            for n_, src in enumerate(real[cid][:12]):
                x0, y0, x1, y1 = res[src.name]["boxes"][0]
                im = cv2.imread(str(src))[int(y0):int(y1), int(x0):int(x1)]
                if im.size and min(im.shape[:2]) >= 16:
                    p = tmpl_dir / f"{cid:03d}_{n_}.png"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(p), im)
                    pngs.append(str(p))
        templates[cid] = pngs

    tasks, rows = [], []
    rnd = random.Random(0)
    for cid in names:
        cname = safe(names[cid])
        if SYN_ONLY and cid not in SYN_ONLY:
            continue
        cdir = OUT / cname
        cdir.mkdir(exist_ok=True)
        keep, extra = ([], []) if SYN_ONLY else pick_diverse(real[cid], TARGET)
        for src in keep:
            link(src, cdir / f"real_{src.name}")
            link(src.with_suffix(".txt"), cdir / f"real_{src.stem}.txt")
        for src in extra:
            link(src, cdir / "extra" / f"real_{src.name}")
            link(src.with_suffix(".txt"), cdir / "extra" / f"real_{src.stem}.txt")
        need = SYN_ONLY[cid] if SYN_ONLY else TARGET - len(keep)
        if need and not templates[cid]:
            rows.append([cid, names[cid], report[cid]["real_found"], len(keep), len(extra), 0,
                         report[cid]["boston"], report[cid]["unlabeled"], "NO TEMPLATE - cannot synthesize"])
            continue
        if need:
            t_aspects = []
            for t in templates[cid]:
                im = cv2.imread(t, cv2.IMREAD_UNCHANGED)
                t_aspects.append(im.shape[1] / im.shape[0])
            pool = [d for d in donors if d["cid"] != cid]
            rnd.shuffle(pool)
            used = 0
            for k in range(need):
                ti = k % len(templates[cid])
                mode = "swap" if rnd.random() < SWAP_SHARE else "place"
                want = t_aspects[ti]
                donor = None
                for _ in range(len(pool)):            # shape-matched donor, never reused for this class
                    d = pool[used % len(pool)]; used += 1
                    if mode == "place" or abs(np.log(d["aspect"] / want)) < np.log(1.35):
                        donor = d; break
                donor = donor or pool[used % len(pool)]
                tasks.append((cid, k, mode, donor, templates[cid][ti], str(cdir / f"syn_{k:05d}.jpg"),
                              cid * 100000 + k + (50000 if SYN_ONLY else 0)))
        rows.append([cid, names[cid], report[cid]["real_found"], len(keep), len(extra), need,
                     report[cid]["boston"], report[cid]["unlabeled"], ""])

    print(f"synthetic images to make: {len(tasks)}", flush=True)
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as pool:
        done = 0
        for r in pool.imap_unordered(make_synthetic, tasks, chunksize=8):
            done += r is not None
            if done % 2000 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)
    print(f"synthetic written: {done}/{len(tasks)}")

    # review sheets: 30 random synthetic per class
    (OUT / "_review").mkdir(exist_ok=True)
    for cid in names:
        syn = sorted((OUT / safe(names[cid])).glob("syn_*.jpg"))
        if not syn:
            continue
        tiles = []
        for p in random.Random(cid).sample(syn, min(30, len(syn))):
            im = cv2.imread(str(p))
            H, W = im.shape[:2]
            c, x, y, w, h = map(float, p.with_suffix(".txt").read_text().split())
            bx0, by0, bx1, by1 = int((x - w / 2) * W), int((y - h / 2) * H), int((x + w / 2) * W), int((y + h / 2) * H)
            m = int(1.5 * max(bx1 - bx0, by1 - by0)) + 10
            crop = im[max(0, by0 - m):by1 + m, max(0, bx0 - m):bx1 + m].copy()
            cv2.rectangle(crop, (bx0 - max(0, bx0 - m), by0 - max(0, by0 - m)),
                          (bx1 - max(0, bx0 - m), by1 - max(0, by0 - m)), (0, 255, 0), 1)
            tiles.append(cv2.resize(crop, (160, 160)))
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 10)
        cv2.imwrite(str(OUT / "_review" / f"{safe(names[cid])}_synthetic.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]))

    with open(OUT / "report.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class_id", "class", "real_confident_SG", "real_used", "real_extra", "synthetic",
                    "boston_excluded", "unlabeled", "note"])
        w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
