"""Extract per-class sign templates from the Singapore sign handbook PDF.

Each table row in the handbook has: number | name | image. The number sits at the
bottom-left of the row, so each image is assigned to the nearest row label whose
baseline is at or below the image's vertical centre.
"""
import json
import re
import shutil
import sys
from pathlib import Path

import fitz  # pymupdf

PDF = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "Singapore Hand Book.pdf"
OUT = Path(__file__).parent / "templates"


def slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def main():
    doc = fitz.open(PDF)
    classes = {}
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir()

    for page in doc:
        # Row labels: lines whose first span is a bare integer (the class id) in the left column.
        rows = []  # (y_bottom_of_row_label, id, name)
        blocks = page.get_text("dict")["blocks"]
        lines = [ln for b in blocks if b["type"] == 0 for ln in b["lines"]]
        num_lines = [ln for ln in lines
                     if re.fullmatch(r"\d{1,3}", "".join(s["text"] for s in ln["spans"]).strip())
                     and ln["bbox"][0] < 100]
        num_lines.sort(key=lambda ln: ln["bbox"][3])
        prev_y = 0
        for nl in num_lines:
            cid = int("".join(s["text"] for s in nl["spans"]).strip())
            y = nl["bbox"][3]
            # Name: text in the name column between the previous row's number and this one.
            name_parts = [(ln["bbox"][1], "".join(s["text"] for s in ln["spans"]).strip())
                          for ln in lines
                          if 95 < ln["bbox"][0] < 230 and prev_y + 3 < ln["bbox"][3] <= y + 3]
            name = " ".join(t for _, t in sorted(name_parts) if t)
            rows.append((y, cid, " ".join(name.split())))
            prev_y = y

        for info in page.get_image_info(xrefs=True):
            x0, y0, x1, y1 = info["bbox"]
            cy = (y0 + y1) / 2
            below = [r for r in rows if r[0] >= cy - 5]
            if not below or not info["xref"]:
                continue
            _, cid, name = below[0]
            pix = fitz.Pixmap(doc, info["xref"])
            if pix.n - pix.alpha >= 4:  # CMYK -> RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
            d = OUT / f"{cid:03d}_{slug(name)}"
            d.mkdir(exist_ok=True)
            k = len(list(d.glob("*.png")))
            pix.save(str(d / f"{k}.png"))
            classes[cid] = name

    classes = dict(sorted(classes.items()))
    (Path(__file__).parent / "classes.json").write_text(json.dumps(classes, indent=1))
    print(f"{len(classes)} classes, {sum(1 for _ in OUT.rglob('*.png'))} template images")
    missing = [i for i in range(1, max(classes) + 1) if i not in classes]
    print("missing ids:", missing)


if __name__ == "__main__":
    main()
