"""Apply the eye-review decisions and cleaning rules -> clean_labels.jsonl (one line per kept image) + clean_counts.csv.
Per image: path, kind, w, h, boxes [[class, cx, cy, w, h]] where class 0-98 = rule-book signs, 99 = hidden Other sign.
Images whose only boxes are dropped are left out (no unlabelled sign is ever learned as background)."""
import csv, json, collections
from pathlib import Path
R = Path(__file__).parent
CM = json.load(open(R / "class_map.json")); C = CM["classes"]
rules = json.load(open(R / "clean_rules.json"))
dec = {}                                             # (path, box index) -> class | "x"
for tag, d in rules["group_decisions"].items():
    pre = {"P": "p", "INFO": "i", "8": "c"}[tag]
    g = json.load(open(R / d / f"{pre}_groups.json"))
    v = dict(l.split() for l in open(R / d / "decisions.txt"))
    for x in g:
        for path, k in x["members"]:
            dec[(path, k)] = v[str(x["group"])]
chk = {(r["path"], int(r["box"])): r["pred"] for r in csv.DictReader(open(R / "check.csv"))}
side = {(r[0], int(r[1])): float(r[4]) for r in csv.reader(open(R / "arrow_side.csv"))}
seen_md5 = set(); out = []; cnt = collections.defaultdict(collections.Counter); drops = collections.Counter()
for l in open(R / "inventory.jsonl"):
    r = json.loads(l)
    if r.get("bad") or r["md5"] in seen_md5:
        drops["duplicate image"] += 1; continue
    seen_md5.add(r["md5"])
    boxes = []; bad_image = False
    for k, (o, n, cx, cy, w, h) in enumerate(r["boxes"]):
        key = (r["path"], k)
        if key in dec:
            n = dec[key]
            if n == "x": drops["review drop"] += 1; continue
            n = int(n)
        if n is None or not isinstance(n, int):
            drops["unmapped"] += 1; continue
        if r["kind"] == "web" and n in rules["drop_web"]:
            bad_image = True; drops[f"web class {n}"] += 1; continue
        if str(n) in rules["drop_check_pred"] and chk.get(key) in rules["drop_check_pred"][str(n)]:
            bad_image = True; drops[f"mirrored {n}"] += 1; continue
        if str(n) in rules["arrow_side_keep"] and r["kind"] == "web":
            s = side.get(key, 0.0)
            if not (s < -0.08 if rules["arrow_side_keep"][str(n)] == "left" else s > 0.08):
                bad_image = True; drops[f"arrow side {n}"] += 1; continue
        if w <= 0 or h <= 0: drops["empty box"] += 1; continue
        boxes.append([n, cx, cy, w, h])
    if bad_image and r["kind"] == "web":
        continue                                     # web picture with a wrong / mirrored sign: drop the whole picture
    if not boxes and r["kind"] != "neg":
        drops["no box left"] += 1; continue
    out.append({"path": r["path"], "kind": r["kind"], "w": r["w"], "h": r["h"], "boxes": boxes})
    for b in boxes: cnt[b[0]][r["kind"]] += 1
with open(R / "clean_labels.jsonl", "w") as fh:
    for o in out: fh.write(json.dumps(o) + "\n")
with open(R / "clean_counts.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["id", "class", "real_boxes", "web_boxes"])
    for i, n in enumerate(C): w.writerow([i, n, cnt[i]["real"], cnt[i]["web"]])
k = collections.Counter(o["kind"] for o in out)
print("kept images", dict(k)); print("dropped", dict(drops))
none = [C[i] for i in range(99) if not cnt[i]["real"] and not cnt[i]["web"]]
print("classes with no real/web image (sign swap only):", len(none), none)
