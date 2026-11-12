"""Post-process extracted templates so every png is exactly one sign (run after extract_templates.py).

SPLIT: pictures that show several variants side by side -> one png per sign (white-gap components)
STACK: an auxiliary plate stored as its own picture -> pasted under each main sign of that class
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

T = Path(__file__).parent / "templates"
SPLIT = ["009", "035", "059", "061", "072/0", "074", "100", "103", "108"]
STACK = {"050": "1", "060": "1", "072": "1"}   # class -> plate png stem


def components(rgb):
    ink = (rgb < 215).any(2).astype(np.uint8)
    grouped = cv2.dilate(ink, np.ones((9, 9), np.uint8))       # letters/arrows inside a sign stay together
    k, lab, st, _ = cv2.connectedComponentsWithStats(grouped)
    if k <= 2:
        return [rgb]
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a > 0.15 * areas.max()]
    keep.sort(key=lambda i: (st[i, cv2.CC_STAT_TOP] // 40, st[i, cv2.CC_STAT_LEFT]))
    out = []
    for i in keep:
        x, y, w, h = st[i, :4]
        x0, y0 = max(0, x + 3), max(0, y + 3)            # undo most of the dilation margin
        out.append(np.ascontiguousarray(rgb[y0:y + h - 3, x0:x + w - 3]))
    return out


def gap_split(rgb, depth=0):
    """Recursively split at a fully blank row/column band in the middle half of the picture."""
    ink = (rgb < 215).any(2)
    for axis in (0, 1):                     # 0: blank rows (stacked signs), 1: blank columns
        prof = ink.any(axis=1 - axis)
        n = len(prof)
        blank = [i for i in range(int(n * 0.25), int(n * 0.75)) if not prof[i]]
        if blank and depth < 3:
            c = blank[len(blank) // 2]
            a, b = (rgb[:c], rgb[c:]) if axis == 0 else (rgb[:, :c], rgb[:, c:])
            return [q for h in (a, b) for q in gap_split(trim(h), depth + 1)]
    return [rgb]


def trim(rgb):
    ys, xs = np.nonzero((rgb < 215).any(2))
    return np.ascontiguousarray(rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


def stack(sign, plate):
    w = sign.shape[1]
    ph = max(1, int(plate.shape[0] * w / plate.shape[1]))
    plate = cv2.resize(plate, (w, ph), interpolation=cv2.INTER_AREA)
    gap = np.full((max(2, w // 30), w, 3), 255, np.uint8)
    return np.vstack([sign, gap, plate])


def cls_dir(prefix):
    return next(T.glob(f"{prefix}_*"))


def main():
    for spec in SPLIT:
        pre, _, stem = spec.partition("/")
        d = cls_dir(pre)
        for p in sorted(d.glob(f"{stem or '*'}.png")):
            parts = [g for c in components(np.array(Image.open(p).convert("RGB"))) for g in gap_split(c)]
            if len(parts) < 2:
                continue
            p.unlink()
            for j, im in enumerate(parts):
                Image.fromarray(im).save(d / f"{p.stem}_{j}.png")
            print(f"{d.name}/{p.name}: split into {len(parts)}")
    for pre, plate_stem in STACK.items():
        d = cls_dir(pre)
        pp = d / f"{plate_stem}.png"
        plate = np.array(Image.open(pp).convert("RGB"))
        pp.unlink()
        for p in sorted(d.glob("*.png")):
            Image.fromarray(stack(np.array(Image.open(p).convert("RGB")), plate)).save(p)
        print(f"{d.name}: plate stacked under {len(list(d.glob('*.png')))} signs")


if __name__ == "__main__":
    main()
