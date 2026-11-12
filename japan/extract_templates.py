"""Extract per-class sign templates from the Japan sign handbook PDF.

Rows are number | name | image(s). Each image goes to the nearest row label whose
baseline is at or below the image's vertical centre (same as the Singapore extractor).
Classes are the 120 folders of `classified Data japan/Merged` (sorted, id 0..119):
  - rows #30 and #77 hold two different signs -> split into two folder classes
  - row #120 repeats #1 -> dropped
Output: templates/NNN_<slug>/*.png with NNN = class id + 1, classes.json {NNN: folder name}

usage: python extract_templates.py <pdf> <merged_folders.txt>
"""
import difflib
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import fitz  # pymupdf
import numpy as np
from PIL import Image

PDF = Path(sys.argv[1])
FOLDERS = [l for l in Path(sys.argv[2]).read_text().splitlines() if l.strip()]
OUT = Path(__file__).parent / "templates"

# handbook row -> folder, per image index (images of a row sorted left->right, top->bottom)
SPLIT = {
    30: ["Handicapped Driver-Passenger", "Handicapped Driver"],              # wheelchair | clover
    77: ["Rest Area - Parking - Fuel Stand", "Rest Area"],                   # P+fuel+food | P+cup
}
SPLIT_AXIS = {30: 1, 77: 0}   # 1 = side by side, 0 = stacked
DROP = {120}


def split_two(rgb, axis):
    """Split at the widest run of blank lines near the middle; crop each half to its sign."""
    ink = (rgb < 200).any(2)
    prof = ink.any(axis=0 if axis == 1 else 1)
    n = len(prof)
    best, cut, run = 0, n // 2, 0
    for i in range(int(n * 0.25), int(n * 0.75)):
        run = run + 1 if not prof[i] else 0
        if run > best:
            best, cut = run, i - run // 2
    halves = [rgb[:, :cut], rgb[:, cut:]] if axis == 1 else [rgb[:cut], rgb[cut:]]
    out = []
    for h in halves:
        m = (h < 200).any(2).astype(np.uint8)
        k, lab, st, _ = cv2.connectedComponentsWithStats(m)
        big = [i for i in range(1, k) if st[i, cv2.CC_STAT_AREA] > 0.02 * m.size]
        ys, xs = np.nonzero(np.isin(lab, big))
        out.append(np.ascontiguousarray(h[ys.min():ys.max() + 1, xs.min():xs.max() + 1]))
    return out


def slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def rows_of(page):
    blocks = page.get_text("dict")["blocks"]
    lines = [ln for b in blocks if b["type"] == 0 for ln in b["lines"]]
    num_lines = [ln for ln in lines
                 if re.fullmatch(r"\d{1,3}", "".join(s["text"] for s in ln["spans"]).strip())
                 and ln["bbox"][0] < 100]
    num_lines.sort(key=lambda ln: ln["bbox"][3])
    rows = []
    for nl in num_lines:
        rid = int("".join(s["text"] for s in nl["spans"]).strip())
        rows.append((nl["bbox"][3], rid))
    return rows


def row_names(doc):
    txt = "".join(p.get_text() for p in doc)
    nums = [(m.start(), int(m.group(1))) for m in re.finditer(r"(?m)^(\d{1,3}) \n", txt)]
    out = {}
    for k, (s, n) in enumerate(nums):
        e = nums[k + 1][0] if k + 1 < len(nums) else len(txt)
        out[n] = " ".join(txt[s:e].split()[1:])
    return out


def main():
    doc = fitz.open(PDF)
    names = row_names(doc)
    by_norm = {norm(f): i for i, f in enumerate(FOLDERS)}
    row_to_cid = {}
    for rid, nm in names.items():
        if rid in DROP or rid in SPLIT:
            continue
        key = norm(nm)
        if key not in by_norm:
            guess = difflib.get_close_matches(key, list(by_norm), n=1, cutoff=0.8)
            if not guess:
                raise SystemExit(f"row {rid} {nm!r} matches no folder")
            print(f"  row {rid} {nm!r} -> {FOLDERS[by_norm[guess[0]]]!r}")
            key = guess[0]
        row_to_cid[rid] = by_norm[key]

    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir()
    per_row = {}
    for page in doc:
        rows = rows_of(page)
        for info in page.get_image_info(xrefs=True):
            x0, y0, x1, y1 = info["bbox"]
            cy = (y0 + y1) / 2
            if not rows or not info["xref"]:
                continue
            # the row number sits at the row's vertical middle
            rid = min(rows, key=lambda r: abs(r[0] - 6 - cy))[1]
            per_row.setdefault(rid, []).append((y0, x0, info["xref"]))

    for rid, imgs in sorted(per_row.items()):
        if rid in DROP:
            continue
        imgs.sort(key=lambda t: (round(t[0] / 40), t[1]))   # rows of images, then left->right
        for _, _, xref in imgs:
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            rgb = arr[..., :3].astype(np.float32)
            if pix.alpha:   # composite transparent parts onto white page
                a = arr[..., 3:4] / 255.0
                rgb = rgb * a + 255 * (1 - a)
            rgb = rgb.astype(np.uint8)
            if rid in SPLIT:   # one picture holding two signs -> split at the widest blank gap
                pieces = split_two(rgb, axis=SPLIT_AXIS[rid])
                targets = [by_norm[norm(f)] for f in SPLIT[rid]]
            else:
                pieces, targets = [rgb], [row_to_cid[rid]]
            for cid, im in zip(targets, pieces):
                d = OUT / f"{cid + 1:03d}_{slug(FOLDERS[cid])}"
                d.mkdir(exist_ok=True)
                Image.fromarray(im).save(d / f"{len(list(d.glob('*.png')))}.png")

    classes = {i + 1: f for i, f in enumerate(FOLDERS)}
    (Path(__file__).parent / "classes.json").write_text(json.dumps(classes, indent=1))
    have = {int(d.name[:3]) for d in OUT.iterdir()}
    print(f"{len(have)} of {len(FOLDERS)} classes have templates, {sum(1 for _ in OUT.rglob('*.png'))} images")
    print("missing:", [classes[i] for i in classes if i not in have])


if __name__ == "__main__":
    main()
