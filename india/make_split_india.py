"""india_train/ for YOLO: train = standard_india (99 classes x 400) + ~5 % sign-free road frames;
val / test = held-out real photos from split.json (never in standard_india). Images are symlinked."""
import json, os, random
from pathlib import Path
R = Path(__file__).parent
C = json.load(open(R / "class_map.json"))["classes"]
D = R / "india_train"
lab = {json.loads(l)["path"]: json.loads(l) for l in open(R / "clean_labels.jsonl")}
split = json.load(open(R / "split.json"))
for s in ("train", "val", "test"):
    for k in ("images", "labels"):
        p = D / k / s; p.mkdir(parents=True, exist_ok=True)
        for f in p.iterdir(): f.unlink()

def link(src, s, name, txt):
    dst = D / "images" / s / name
    os.symlink(src, dst)
    (D / "labels" / s / (Path(name).stem + ".txt")).write_text(txt)

n = 0
for d in sorted((R / "standard_india").iterdir()):
    if not d.is_dir(): continue
    for im in sorted(d.glob("*.jpg")):
        link(im, "train", f"{d.name[:3]}_{im.name}", im.with_suffix(".txt").read_text()); n += 1
negs = [p for p, r in lab.items() if r["kind"] == "neg"]
random.Random(3).shuffle(negs)
for p in negs[: n // 20]:
    link(p, "train", "neg_" + Path(p).name, "")
for s in ("val", "test"):
    for p in split[s]:
        r = lab[p]
        link(p, s, Path(p).name, "".join(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n" for b in r["boxes"]))
(D / "data.yaml").write_text(f"path: {D}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: {len(C)}\nnames:\n"
                             + "".join(f"  {i}: {json.dumps(c)}\n" for i, c in enumerate(C)))
print({s: len(list((D / "images" / s).iterdir())) for s in ("train", "val", "test")})
