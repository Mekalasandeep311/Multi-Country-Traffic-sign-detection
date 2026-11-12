"""Find the nuScenes city (Boston n008 / Singapore n015) of renamed 1600x900 frames by matching
them to the original nuScenes images with a 64-bit difference hash.

Searches CAM_FRONT keyframes first, then CAM_FRONT sweeps, then the other cameras' keyframes.
Writes nusc_city.csv: class,file,city,nuscenes_file,hamming

usage: python nusc_city.py
"""
import csv
import glob
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "Traffic Signs Singapore"
NUSC = Path("/data/common_datasets/nuScenes")
MAX_HAMMING = 8


def dhash(path):
    im = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if im is None:
        return path, None
    small = cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return path, int(np.packbits(bits).view(">u8")[0])


def hashes(paths, pool):
    return {p: h for p, h in pool.imap_unordered(dhash, paths, chunksize=256) if h is not None}


def popcount(x):
    return bin(x).count("1")


def main():
    queries = []
    for f in glob.glob(str(DATA / "*" / "*.jpg")):
        im = cv2.imread(f, cv2.IMREAD_REDUCED_GRAYSCALE_8)
        if im is not None and im.shape[:2] == (900 // 8 + (900 % 8 > 0), 1600 // 8):
            queries.append(f)
    print("nuScenes-size frames:", len(queries), flush=True)

    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as pool:
        q = hashes(queries, pool)
        found = {}
        stages = [NUSC / "samples" / "CAM_FRONT", NUSC / "sweeps" / "CAM_FRONT"] + \
                 [NUSC / "samples" / c for c in ("CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK",
                                                 "CAM_BACK_LEFT", "CAM_BACK_RIGHT")]
        for stage in stages:
            todo = {p: h for p, h in q.items() if p not in found}
            if not todo:
                break
            ref = hashes(sorted(glob.glob(str(stage / "*.jpg"))), pool)
            ref_h = np.array(list(ref.values()), dtype=np.uint64)
            ref_p = list(ref.keys())
            for p, h in todo.items():
                d = np.array([popcount(int(x)) for x in np.bitwise_xor(ref_h, np.uint64(h))])
                i = int(d.argmin())
                if d[i] <= MAX_HAMMING:
                    found[p] = (ref_p[i], int(d[i]))
            print(f"{stage.parent.name}/{stage.name}: matched {len(found)}/{len(q)}", flush=True)

    with open(ROOT / "nusc_city.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "file", "city", "nuscenes_file", "hamming"])
        for p in sorted(q):
            ref, d = found.get(p, ("", -1))
            name = os.path.basename(ref)
            city = "boston" if name.startswith("n008") else "singapore" if name.startswith("n015") else "unknown"
            w.writerow([Path(p).parent.name, Path(p).name, city, name, d])
    print("wrote", ROOT / "nusc_city.csv")


if __name__ == "__main__":
    main()
