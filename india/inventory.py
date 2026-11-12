"""Inventory of the two read-only source datasets -> inventory.jsonl (one line per image).
Fields: stem, path, kind (real = camera photo; web = scraped/generated picture; neg = sign-free road image),
w, h, md5, dhash (near-duplicate check), boxes [[old_id, new_id_or_tag, cx, cy, w, h], ...]."""
import hashlib, json, os, re, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
RAW = Path("/data/common_datasets/traffic_signs_india")
SPLIT = Path("/data/common_datasets/traffic_signs_india_split")
R = Path(__file__).parent
M = json.load(open(R / "class_map.json"))["old2new"]
IMG = (".jpg", ".jpeg", ".png")

def kind(stem):
    if stem.isdigit() or stem.lower().startswith("whatsapp"):
        return "real"
    return "web"

def dhash(g):
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    return int("".join("1" if a > b else "0" for a, b in zip(s[:, 1:].ravel(), s[:, :-1].ravel())), 2)

def one(args):
    path, txt, k = args
    data = open(path, "rb").read()
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return {"stem": path.stem, "path": str(path), "kind": k, "bad": True}
    boxes = []
    if txt and os.path.exists(txt):
        for l in open(txt):
            p = l.split()
            if len(p) >= 5:
                o = int(float(p[0]))
                boxes.append([o, M.get(str(o)), *map(float, p[1:5])])
    return {"stem": path.stem, "path": str(path), "kind": k, "w": im.shape[1], "h": im.shape[0],
            "md5": hashlib.md5(data).hexdigest(), "dhash": dhash(im), "boxes": boxes}

jobs = []
for n in sorted(os.listdir(RAW)):
    s, e = os.path.splitext(n)
    if e.lower() in IMG:
        jobs.append((RAW / n, str(RAW / (s + ".txt")), kind(s)))
for n in sorted(os.listdir(SPLIT / "images" / "train")):
    s, e = os.path.splitext(n)
    if s.startswith("min") and e.lower() in IMG:
        jobs.append((SPLIT / "images" / "train" / n, str(SPLIT / "labels" / "train" / (s + ".txt")), "neg"))
print("images", len(jobs), flush=True)
with open(R / "inventory.jsonl", "w") as fh, ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex:
    for i, r in enumerate(ex.map(one, jobs, chunksize=64)):
        fh.write(json.dumps(r) + "\n")
        if i % 5000 == 0:
            print(i, flush=True)
print("done")
