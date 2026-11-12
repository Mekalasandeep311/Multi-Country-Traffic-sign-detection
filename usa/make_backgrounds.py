"""Pick sign-background frames for the synthetic stage and cache OWLv2 sign boxes on them.

Frames come from the US drive folders (video*/frame_*.jpg), picked evenly over all drives so scenes
vary. backgrounds/<drive>__<name>.jpg (hard link) + backgrounds/cache/<stem>.json (boxes that
make_yolo_dataset.py inpaints before pasting templates).

usage: python make_backgrounds.py [n]
"""
import json
import os
import sys
from pathlib import Path

from PIL import Image

from autolabel import SignDetector

ROOT = Path(__file__).parent
OUT = ROOT / "backgrounds"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
SKIP = 300            # first/last frames of a drive often carry YouTube intro/outro text


def main():
    pool = []
    for d in sorted(p for p in ROOT.glob("video*") if p.is_dir()):
        fs = sorted(d.glob("frame_*.jpg"))
        pool += fs[SKIP:-SKIP]
    pick = [pool[int(i * len(pool) / N)] for i in range(N)]
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    det = SignDetector()
    for k, p in enumerate(pick):
        name = f"{p.parent.name}__{p.name}"
        dst = OUT / name
        if not dst.exists():
            os.link(p, dst)
        cache = OUT / "cache" / f"{Path(name).stem}.json"
        if not cache.exists():
            cache.write_text(json.dumps(det(Image.open(p).convert("RGB"))))
        if k % 50 == 0:
            print(f"{k}/{N}", flush=True)
    print(f"{len(pool)} frames in the pool, {N} backgrounds in {OUT}")


if __name__ == "__main__":
    main()
