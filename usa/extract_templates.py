"""Extract per-class sign templates from the USA road sign handbook PDF.

Rows are number | name | image(s), same layout as the Japan handbook. Every row is one class:
class id = row number - 1 (0..208), class name = the row's sign name.
Each image goes to the row whose number sits nearest the image's vertical centre.
Output: templates/NNN_<slug>/*.png (NNN = row number), classes.json {NNN: name}, classes.txt

usage: python extract_templates.py <pdf>
"""
import json
import re
import shutil
import sys
from pathlib import Path

import fitz  # pymupdf
import numpy as np
from PIL import Image

PDF = Path(sys.argv[1])
ROOT = Path(__file__).parent
OUT = ROOT / "templates"


def slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def rows_of(page):
    blocks = page.get_text("dict")["blocks"]
    lines = [ln for b in blocks if b["type"] == 0 for ln in b["lines"]]
    num_lines = [ln for ln in lines
                 if re.fullmatch(r"\d{1,3}", "".join(s["text"] for s in ln["spans"]).strip())
                 and ln["bbox"][0] < 100]
    return sorted(((ln["bbox"][3], int("".join(s["text"] for s in ln["spans"]).strip())) for ln in num_lines))


def row_names(doc):
    txt = "".join(p.get_text() for p in doc)
    nums = [(m.start(), int(m.group(1))) for m in re.finditer(r"(?m)^(\d{1,3}) \n", txt)]
    out = {}
    for k, (s, n) in enumerate(nums):
        e = nums[k + 1][0] if k + 1 < len(nums) else len(txt)
        body = txt[s:e]
        body = re.split(r"\n\s*\d+\. [A-Z]", body)[0]          # cut a following section title
        body = body.replace("No. \nSign name \nSign", "")
        out[n] = " ".join(body.split()[1:])
    return out


def main():
    doc = fitz.open(PDF)
    names = row_names(doc)
    nrows = max(names)
    missing_names = [i for i in range(1, nrows + 1) if i not in names]
    if missing_names:
        raise SystemExit(f"rows without a name: {missing_names}")
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir()
    per_row = {}
    for page in doc:
        rows = rows_of(page)
        for info in page.get_image_info(xrefs=True):
            x0, y0, x1, y1 = info["bbox"]
            if not rows or not info["xref"]:
                continue
            rid = min(rows, key=lambda r: abs(r[0] - 6 - (y0 + y1) / 2))[1]
            per_row.setdefault(rid, []).append((y0, x0, info["xref"]))
    for rid, imgs in sorted(per_row.items()):
        imgs.sort(key=lambda t: (round(t[0] / 40), t[1]))
        d = OUT / f"{rid:03d}_{slug(names[rid])}"
        d.mkdir(exist_ok=True)
        for k, (_, _, xref) in enumerate(imgs):
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            rgb = arr[..., :3].astype(np.float32)
            if pix.alpha:
                a = arr[..., 3:4] / 255.0
                rgb = rgb * a + 255 * (1 - a)
            Image.fromarray(rgb.astype(np.uint8)).save(d / f"{k}.png")
    classes = {i: names[i] for i in range(1, nrows + 1)}
    (ROOT / "classes.json").write_text(json.dumps(classes, indent=1, ensure_ascii=False))
    (ROOT / "classes.txt").write_text("".join(f"{classes[i]}\n" for i in range(1, nrows + 1)))
    have = {int(d.name[:3]) for d in OUT.iterdir()}
    print(f"{nrows} classes, {len(have)} with templates, {sum(1 for _ in OUT.rglob('*.png'))} images")
    print("no template:", [f"{i} {classes[i]}" for i in classes if i not in have])
    print("several images:", [d.name for d in sorted(OUT.iterdir()) if len(list(d.glob('*.png'))) > 1])


if __name__ == "__main__":
    main()
