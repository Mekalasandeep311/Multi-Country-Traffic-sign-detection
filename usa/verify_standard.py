"""Check standard_usa/: every class folder has exactly TARGET image + .txt pairs with valid YOLO labels.

Checks per class folder (top level, not extra/): image count, each image has a .txt, every line is
"cid cx cy w h" with cid in range and coordinates in (0, 1], the first line is the folder's own class,
synthetic .txt has exactly one line. Prints a summary and every problem found.

usage: python verify_standard.py [folder] [target]
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "standard_usa"
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 400


def safe(name):
    return name.replace("/", "-").replace(":", "-").strip()


def main():
    names = (OUT / "classes.txt").read_text().splitlines()
    problems, kinds = [], Counter()
    for cid, name in enumerate(names):
        d = OUT / safe(name)
        imgs = sorted(d.glob("*.jpg"))
        if len(imgs) != TARGET:
            problems.append(f"{name}: {len(imgs)} images")
        for im in imgs:
            kind = "syn" if im.name.startswith("syn_") else "mtsd" if im.name.startswith("real_mtsd_") else "drive"
            kinds[kind] += 1
            t = im.with_suffix(".txt")
            if not t.exists():
                problems.append(f"{name}/{im.name}: no .txt")
                continue
            lines = t.read_text().split("\n")
            lines = [l for l in lines if l.strip()]
            if not lines:
                problems.append(f"{name}/{t.name}: empty")
                continue
            if kind == "syn" and len(lines) != 1:
                problems.append(f"{name}/{t.name}: synthetic with {len(lines)} lines")
            for i, l in enumerate(lines):
                v = l.split()
                ok = len(v) == 5 and v[0].isdigit() and int(v[0]) < len(names)
                if ok:
                    x, y, w, h = map(float, v[1:])
                    ok = 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
                if not ok:
                    problems.append(f"{name}/{t.name}: bad line {l!r}")
                elif i == 0 and int(v[0]) != cid:
                    problems.append(f"{name}/{t.name}: first label is class {v[0]}, not {cid}")
    print(f"{len(names)} classes; images: drive {kinds['drive']}, MTSD {kinds['mtsd']}, synthetic {kinds['syn']}")
    print(f"{len(problems)} problems")
    for p in problems[:50]:
        print("  ", p)


if __name__ == "__main__":
    main()
