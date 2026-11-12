"""Synthetic "Other sign (not in list)" class (id 209) for the USA model.

The video test (clip2) showed the 209-class model forcing US text plates that are NOT among its classes
(NO TURNS, BUSES ONLY, LEFT TURN SIGNAL ...) onto the nearest class, often at conf 0.9+. This script renders
such plates (phrases checked against classes.txt: no text of a real class) and puts them into real US drive
frames with the same sign-swap machinery as build_balanced_dataset.py:
  swap    a real sign is replaced by the plate (plates stand where signs stand)
  attach  the real sign is kept (labelled with its class) and the plate is stacked under/over it, as in NYC
  place   plate on a drawn pole at a random spot
Every other sign-like box of the donor frame is inpainted, so the labels are complete.

Output: other_sign/Other sign (not in list)/syn_other_<n>.jpg + .txt, other_sign/_review_*.jpg
usage: python make_other_signs.py [count]
"""
import glob
import json
import os
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

_argv, sys.argv = sys.argv, sys.argv[:1]           # build_balanced_dataset reads sys.argv on import
import build_balanced_dataset as bb                 # noqa: E402
sys.argv = _argv

ROOT = Path(__file__).parent
COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
OTHER_ID = 209
OTHER_NAME = "Other sign (not in list)"
OUT = ROOT / "other_sign"
MODES = (("swap", 0.5), ("attach", 0.3), ("place", 0.2))

# texts of US signs that are NOT among the 209 classes (no speed limit / parking / standing / stopping /
# one way / do not enter / stop here on red / no turn on red / road work ahead / road closed / school texts)
WHITE = ["NO TURNS", "NO TURNS|8AM-6PM", "BUSES ONLY", "BUSES|ONLY", "BUS LANE", "BUSES ONLY|7AM-7PM|MON-FRI",
         "LEFT TURN|SIGNAL", "LEFT TURN|YIELD|ON GREEN", "LEFT LANE|MUST|TURN LEFT", "RIGHT LANE|MUST|TURN RIGHT",
         "ONLY", "THRU|TRAFFIC", "ALL TRAFFIC", "EXCEPT|BUSES", "EXCEPT|BICYCLES", "LOCAL|TRAFFIC|ONLY",
         "TRUCK|ROUTE", "NO TRUCKS", "NO TRUCKS|OVER 33 FEET", "TRUCKS OVER|33 FEET|PROHIBITED", "NO THRU|TRAFFIC",
         "DEAD END", "NO OUTLET", "NOT A|THROUGH|STREET", "DON'T|BLOCK|THE BOX", "DON'T BLOCK|THE BOX|$350 FINE",
         "NO|HONKING", "$350 FINE", "TRAFFIC LAWS|PHOTO|ENFORCED", "BEGIN", "END", "AHEAD", "THIS BLOCK",
         "EXCEPT|SUNDAY", "MON-FRI", "7AM-7PM", "8AM-6PM", "4PM-7PM|MON-FRI", "LOADING|ZONE",
         "COMMERCIAL|VEHICLES|ONLY", "AUTHORIZED|VEHICLES|ONLY", "EMERGENCY|VEHICLES|ONLY", "FIRE LANE",
         "TAXI|STAND", "YIELD|TO BUSES", "WAIT|HERE", "RIGHT TURN|ON RED|AFTER STOP", "LEFT TURNS|PROHIBITED",
         "QUIET ZONE", "HOSPITAL|ZONE", "NO ENGINE|BRAKE", "BRIDGE ICES|BEFORE|ROAD", "PRIVATE|PROPERTY",
         "NO|TRESPASSING", "USE OTHER|SIDEWALK", "LEFT TURN|ON ARROW|ONLY", "THRU TRAFFIC|MERGE LEFT",
         "LANE ENDS|MERGE|LEFT", "BIKE|ROUTE", "END|DETOUR",]
WHITE_ARROWS = ["←", "→", "↑", "← →", "↖", "↗"]
RED_TEXT = ["NO|HONKING", "FIRE|LANE", "EMERGENCY|SNOW ROUTE", "NO|TRESPASSING", "AUTHORIZED|VEHICLES|ONLY"]
RED_BG = ["WRONG|WAY", "WRONG WAY", "EMERGENCY|SNOW|ROUTE"]
BLUE = ["H", "HOSPITAL", "P", "PARKING", "EMERGENCY", "TAXI", "GAS|FOOD|LODGING", "INFO", "BUS"]
YELLOW = ["DIP", "BUMP", "ROUGH|ROAD", "LOW|SHOULDER", "SOFT|SHOULDER", "PAVEMENT|ENDS", "DEAD|END",
          "NO|OUTLET", "SLOW", "CAUTION", "WATCH FOR|TRUCKS", "BLIND|DRIVEWAY", "HIDDEN|DRIVE", "NARROW|BRIDGE",
          "ONE LANE|BRIDGE", "SHARP|CURVE", "USE LOW|GEAR", "LOOSE|GRAVEL", "FLOOD|AREA", "PED|XING", "XING",
          "CROSS TRAFFIC|DOES NOT STOP", "NEXT 2|MILES", "WHEN|FLASHING"]
ORANGE = ["DETOUR", "LANE|CLOSED", "RIGHT LANE|CLOSED", "LEFT LANE|CLOSED", "SIDEWALK|CLOSED", "FLAGGER|AHEAD",
          "END|ROAD WORK", "UTILITY|WORK|AHEAD", "SHOULDER|WORK", "ONE LANE|ROAD|AHEAD", "UNEVEN|LANES",
          "FRESH|OIL", "DANGER", "DANGER|HIGH PRESSURE|TESTING", "TRUCKS|ENTERING", "WORKERS|AHEAD"]
STYLES = [  # (weight, phrases, background, text/border colour, shape)
    (0.55, WHITE, (245, 245, 242), (20, 20, 20), "rect"),
    (0.07, RED_TEXT, (245, 245, 242), (200, 30, 35), "rect"),
    (0.05, RED_BG, (200, 30, 35), (245, 245, 242), "rect"),
    (0.06, BLUE, (20, 70, 160), (245, 245, 242), "rect"),
    (0.15, YELLOW, (250, 205, 20), (20, 20, 20), "diamond_or_rect"),
    (0.12, ORANGE, (245, 120, 20), (20, 20, 20), "diamond_or_rect"),
]


def font_path():
    for pat in ("**/DejaVuSans-Bold.ttf", "**/LiberationSans-Bold.ttf", "**/*Bold*.ttf"):
        hits = glob.glob(os.path.join(sys.prefix, pat), recursive=True) or glob.glob(f"/usr/share/fonts/{pat}", recursive=True)
        if hits:
            return hits[0]
    raise SystemExit("no bold TTF font found")


FONT = None


def render_plate(rng):
    """RGBA uint8 template (RGB order) of a random out-of-class US plate."""
    w = np.array([s[0] for s in STYLES])
    _, phrases, bg, fg, shape = STYLES[rng.choice(len(STYLES), p=w / w.sum())]
    text = phrases[rng.integers(len(phrases))]
    lines = text.split("|")
    if phrases is WHITE and rng.random() < 0.25:
        lines.append(WHITE_ARROWS[rng.integers(len(WHITE_ARROWS))])
    if shape == "diamond_or_rect":
        shape = "diamond" if len(lines) <= 2 and max(map(len, lines)) <= 9 and rng.random() < 0.7 else "rect"
    fsize = 64
    font = ImageFont.truetype(FONT, fsize)
    hx = rng.uniform(0.72, 1.0)                    # condensed lettering like Highway Gothic
    sizes = [font.getbbox(l) for l in lines]
    lw = [int((b[2] - b[0]) * hx) for b in sizes]
    lh = int(fsize * 1.15)
    tw, th = max(lw), lh * len(lines)
    if shape == "diamond":
        S = int(max(tw, th) * 1.9) + 20
        W, H = S, S
    else:
        padx, pady = int(fsize * rng.uniform(0.35, 0.7)), int(fsize * rng.uniform(0.3, 0.55))
        W, H = tw + 2 * padx, th + 2 * pady
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bw = max(3, int(min(W, H) * 0.035))
    if shape == "diamond":
        poly = [(W / 2, 0), (W - 1, H / 2), (W / 2, H - 1), (0, H / 2)]
        d.polygon(poly, fill=bg + (255,))
        k = 0.06 * W
        d.polygon([(W / 2, k), (W - 1 - k, H / 2), (W / 2, H - 1 - k), (k, H / 2)], outline=fg + (255,), width=bw)
    else:
        r = int(min(W, H) * 0.08)
        d.rounded_rectangle([0, 0, W - 1, H - 1], r, fill=bg + (255,))
        if rng.random() < 0.85:
            k = bw * 1.2
            d.rounded_rectangle([k, k, W - 1 - k, H - 1 - k], max(1, r - int(k)), outline=fg + (255,), width=bw)
    y = (H - th) / 2
    for l, b, wl in zip(lines, sizes, lw):
        layer = Image.new("RGBA", (b[2] - b[0] + 4, lh), (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((-b[0] + 2, (lh - fsize) / 2 - 2), l, font=font, fill=fg + (255,))
        layer = layer.resize((max(1, wl), lh))
        img.alpha_composite(layer, (int((W - wl) / 2), int(y)))
        y += lh
    return np.array(img)


def make_other(task):
    k, donor, mode, out_img, seed = task
    rng = np.random.default_rng(seed)
    img = cv2.cvtColor(cv2.imread(donor["path"]), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    x0, y0, x1, y1 = donor["box"]
    donor_crop = img[int(y0):int(y1), int(x0):int(x1)].copy()
    erase = donor["others"] + ([] if mode == "attach" else [donor["box"]])
    img = bb.inpaint_boxes(img, erase) if erase else img
    labels = []
    if mode == "attach":
        labels.append((donor["cid"], (x0, y0, x1, y1)))
    n_plates = 2 if (mode != "place" and rng.random() < 0.25) else 1
    bh = y1 - y0
    cx = (x0 + x1) / 2
    below = rng.random() < 0.75
    edge = y1 if below else y0
    for j in range(n_plates):
        tmpl = render_plate(rng)
        if mode == "swap" and j == 0:
            h = max(12, int(round(bh * rng.uniform(0.8, 1.2))))
            cy = (y0 + y1) / 2
        elif mode == "place":
            h = int(np.exp(rng.uniform(np.log(18), np.log(150))))
        else:                                      # stacked under/over the sign (or the previous plate)
            h = max(10, int(round(bh * rng.uniform(0.35, 0.9))))
        sign = bb.warp_sign(tmpl, h, rng)
        sign = bb.match_appearance(sign, donor_crop, rng)
        sh, sw = sign.shape[:2]
        if mode == "place":
            cx = rng.uniform(0.05 * W + sw / 2, 0.95 * W - sw / 2) if 0.9 * W > sw else W / 2
            cy = rng.uniform(0.25 * H, 0.6 * H)
            bb.draw_pole(img, cx, cy + sh / 2 - 2, sh, rng)
        elif not (mode == "swap" and j == 0):
            gap = rng.uniform(0, 0.08) * sh
            cy = edge + gap + sh / 2 if below else edge - gap - sh / 2
        if cy - sh / 2 < 0 or cy + sh / 2 > H:
            continue
        box = bb.paste(img, sign, cx, cy)
        if box is None:
            continue
        labels.append((OTHER_ID, box))
        edge = box[3] if below else box[1]
    if not any(c == OTHER_ID for c, _ in labels):
        return None
    cv2.imwrite(out_img, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(85, 96))])
    Path(out_img).with_suffix(".txt").write_text("".join(
        f"{c} {(a + b) / 2 / W:.6f} {(e + f) / 2 / H:.6f} {(b - a) / W:.6f} {(f - e) / H:.6f}\n"
        for c, (a, e, b, f) in labels))
    return out_img


def donors_list():
    """Accepted US drive signs usable as donors (same rule as build_balanced_dataset.main)."""
    cmap = yaml.safe_load((ROOT / "class_map.yaml").read_text())
    out = []
    for folder, cid in cmap["folders"].items():
        res_file = bb.AUTO / f"{folder}.json"
        if not res_file.exists():
            continue
        res = json.loads(res_file.read_text())
        cache = np.load(bb.AUTO / "cache" / f"{folder}.npz")
        files = list(cache["files"])
        for f, r in res.items():
            if r["status"] != "accepted":
                continue
            b = r["boxes"][0]
            if 12 <= b[3] - b[1] <= bb.MAX_DONOR_H:
                i = files.index(f)
                sel = (cache["img_idx"] == i) & (cache["sign"] >= bb.OTHER_SIGN)
                others = [bx.tolist() for bx in cache["boxes"][sel]] + [bx for bx in r["boxes"][1:]]
                # attach mode keeps the donor sign: never erase boxes that overlap it
                others = [o for o in others if bb_iou(o, b) < 0.3]
                out.append({"path": str(bb.DATA / folder / f), "box": b, "others": others, "cid": cid})
    return out


def bb_iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih)


def init():
    global FONT
    FONT = font_path()


def main():
    init()
    print("font", FONT)
    donors = donors_list()
    print(f"donor signs: {len(donors)}", flush=True)
    d = OUT / bb.safe(OTHER_NAME)
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)                         # only our own output lives here
    d.mkdir(parents=True)
    rnd = random.Random(209)
    rnd.shuffle(donors)
    names, p = [m for m, _ in MODES], [w for _, w in MODES]
    tasks = [(k, donors[k % len(donors)], rnd.choices(names, p)[0], str(d / f"syn_other_{k:05d}.jpg"), 209_000 + k)
             for k in range(int(COUNT * 1.1))]      # a few fail (plate outside the frame)
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)), initializer=init) as pool:
        made = [r for r in pool.imap(make_other, tasks, chunksize=8) if r]
    for extra in made[COUNT:]:                     # keep exactly COUNT
        Path(extra).unlink(); Path(extra).with_suffix(".txt").unlink()
    made = made[:COUNT]
    print(f"written {len(made)} images", flush=True)
    rng = random.Random(1)
    for s in range(2):                             # review sheets with boxes drawn
        tiles = []
        for p_ in rng.sample(made, 24):
            im = cv2.imread(p_)
            Hh, Ww = im.shape[:2]
            for l in Path(p_).with_suffix(".txt").read_text().split("\n"):
                if l:
                    c, x, y, w, h = map(float, l.split())
                    col = (0, 0, 255) if int(c) == OTHER_ID else (0, 255, 0)
                    cv2.rectangle(im, (int((x - w / 2) * Ww), int((y - h / 2) * Hh)),
                                  (int((x + w / 2) * Ww), int((y + h / 2) * Hh)), col, 2)
                    cx_, cy_ = int(x * Ww), int(y * Hh)
            m = 200
            crop = im[max(0, cy_ - m):cy_ + m, max(0, cx_ - m):cx_ + m]
            tiles.append(cv2.resize(crop, (300, 300)))
        cv2.imwrite(str(OUT / f"_review_{s}.jpg"), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, 24, 6)]))


if __name__ == "__main__":
    main()
