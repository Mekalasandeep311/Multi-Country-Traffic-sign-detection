"""Leakage of the OLD split: old test/val images with an exact (md5) or near (dHash <= 6) copy in old train."""
import json, os
from pathlib import Path
S = "/data/common_datasets/traffic_signs_india_split/images"
inv = {Path(json.loads(l)["path"]).stem: json.loads(l) for l in open("inventory.jsonl")}
sets = {s: {Path(n).stem for n in os.listdir(f"{S}/{s}")} for s in ("train", "val", "test")}
tr = [inv[s] for s in sets["train"] if s in inv and not inv[s].get("bad")]
md5 = {r["md5"] for r in tr}; hs = [r["dhash"] for r in tr]
import numpy as np
H = np.array(hs, dtype=np.uint64)
def near(h):
    x = np.bitwise_xor(H, np.uint64(h)); return int(min(bin(int(v)).count("1") for v in x[np.argsort(x)[:0]] ) if False else 0)
for s in ("test", "val"):
    rs = [inv[k] for k in sets[s] if k in inv and not inv[k].get("bad")]
    ex = sum(r["md5"] in md5 for r in rs); nr = 0
    for r in rs:
        x = np.bitwise_xor(H, np.uint64(r["dhash"]))
        d = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1).min()
        nr += d <= 6
    print(f"old {s}: {len(rs)} images, exact copy in train {ex} ({ex/len(rs):.1%}), near copy (dHash<=6) {nr} ({nr/len(rs):.1%})")
