"""Pick the real frames to auto-label: every STEP-th frame of each US drive + all Boston frames.

Drops the first/last SKIP frames of a drive (YouTube intro/outro text) and near-duplicates
(difference hash within HAM bits of the last kept frame of the same drive, e.g. waiting at a red light).
Writes real/frames.csv: path, source, kept (1/0), reason.

usage: python make_frame_pool.py [step]
"""
import csv
import sys
from pathlib import Path

from build_balanced_dataset import dhash

ROOT = Path(__file__).parent
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SKIP = 300
HAM = 4


def main():
    rows = []
    for d in sorted(p for p in ROOT.glob("video*") if p.is_dir()):
        fs = sorted(d.glob("frame_*.jpg"))
        last = None
        for i, p in enumerate(fs):
            if i % STEP:
                continue
            if i < SKIP or i >= len(fs) - SKIP:
                rows.append([str(p.relative_to(ROOT)), d.name, 0, "intro/outro"])
                continue
            h = dhash(p)
            if last is not None and bin(h ^ last).count("1") <= HAM:
                rows.append([str(p.relative_to(ROOT)), d.name, 0, "near-duplicate"])
                continue
            last = h
            rows.append([str(p.relative_to(ROOT)), d.name, 1, ""])
        print(d.name, sum(r[2] for r in rows if r[1] == d.name), flush=True)
    for p in sorted((ROOT / "boston").glob("*.jpg")):
        rows.append([str(p.relative_to(ROOT)), "boston", 1, ""])
    (ROOT / "real").mkdir(exist_ok=True)
    with open(ROOT / "real" / "frames.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "source", "kept", "reason"])
        w.writerows(rows)
    kept = sum(r[2] for r in rows)
    print(f"{len(rows)} frames looked at, {kept} kept")


if __name__ == "__main__":
    main()
