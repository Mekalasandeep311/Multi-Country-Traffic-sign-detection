"""Pick sign-background frames for the synthetic stage and cache OWLv2 sign boxes on them.

Frames come from the raw sequence `Japan/Classify_Split` and are NOT in any Merged class folder, picked
evenly over the whole drive so scenes vary. backgrounds/<name>.jpg (hard link) + backgrounds/cache/<stem>.json
(boxes that make_yolo_dataset.py inpaints before pasting templates).

usage: python make_backgrounds.py [n]
"""
import json
import os
import re
import sys
from pathlib import Path

from PIL import Image

from autolabel import SignDetector

ROOT = Path(__file__).parent
CDJ = ROOT / "classified Data japan"
RAW = CDJ / "Japan" / "Classify_Split"
OUT = ROOT / "backgrounds"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400


def key(p):
    m = re.fullmatch(r"(N?)(\d+)", p.stem)
    return (m.group(1), int(m.group(2))) if m else ("~", 0)


def main():
    classified = {re.sub(r"_\d+$", "", p.stem) for p in (CDJ / "Merged").rglob("*.jpg")}
    pool = sorted((p for p in RAW.rglob("*.jpg") if p.stem not in classified), key=key)
    pick = [pool[int(i * len(pool) / N)] for i in range(N)]
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    det = SignDetector()
    for k, p in enumerate(pick):
        dst = OUT / p.name
        if not dst.exists():
            os.link(p, dst)
        cache = OUT / "cache" / f"{p.stem}.json"
        if not cache.exists():
            cache.write_text(json.dumps(det(Image.open(p).convert("RGB"))))
        if k % 50 == 0:
            print(f"{k}/{N}", flush=True)
    print(f"{len(pool)} unclassified frames, {N} backgrounds in {OUT}")


if __name__ == "__main__":
    main()
