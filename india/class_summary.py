"""Per new class: real / web images, unique after near-duplicate grouping (dHash Hamming <= 6)."""
import json, collections
from pathlib import Path
R = Path(__file__).parent
C = json.load(open(R / "class_map.json"))["classes"]
rows = [json.loads(l) for l in open(R / "inventory.jsonl")]
seen_md5 = set(); groups = {}
def near(h):
    for g in groups:
        if bin(g ^ h).count("1") <= 6:
            return g
    return None
stat = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    if r.get("bad") or r["md5"] in seen_md5:
        continue
    seen_md5.add(r["md5"])
    tags = {b[1] for b in r["boxes"]}
    for t in tags:
        key = t if t is not None else "drop"
        stat[key][r["kind"]] += 1
out = []
for i, n in enumerate(C):
    s = stat.get(i, {})
    out.append((i, n, s.get("real", 0), s.get("web", 0)))
for t in ("P", "INFO", "drop"):
    print(t, dict(stat.get(t, {})))
print("id name real web")
for i, n, a, b in out:
    print(f"{i:3d} {n[:38]:38s} {a:5d} {b:5d}")
json.dump(out, open(R / "class_summary.json", "w"))
