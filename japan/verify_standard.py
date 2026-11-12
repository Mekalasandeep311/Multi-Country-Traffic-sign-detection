from pathlib import Path
B = Path("standard_japan")
names = [l for l in (B / "classes.txt").read_text().splitlines() if l]
dirs = [d for d in B.iterdir() if d.is_dir() and not d.name.startswith("_")]
bad, counts = [], {}
for d in dirs:
    imgs = list(d.glob("*.jpg"))
    counts[d.name] = len(imgs)
    for im in imgs + list((d / "extra").glob("*.jpg")):
        t = im.with_suffix(".txt")
        if not t.exists():
            bad.append(f"no txt {im}"); continue
        lines = [l.split() for l in t.read_text().splitlines() if l.strip()]
        if not lines: bad.append(f"empty {t}")
        if im.name.startswith("syn_") and len(lines) != 1: bad.append(f"syn lines {len(lines)} {t}")
        for l in lines:
            c, *v = l
            if names[int(c)].replace("/", "-").strip() != d.name: bad.append(f"class {c} in {d.name}")
            if not all(0 <= float(x) <= 1 for x in v): bad.append(f"coords {t}")
print("class folders", len(dirs), "| classes.txt", len(names))
print("folders with exactly 400:", sum(v == 400 for v in counts.values()), "| others:", {k: v for k, v in counts.items() if v != 400})
print("problems:", len(bad), bad[:5])
print("total images (excl extra):", sum(counts.values()), "| extra:", len(list(B.glob("*/extra/*.jpg"))))
