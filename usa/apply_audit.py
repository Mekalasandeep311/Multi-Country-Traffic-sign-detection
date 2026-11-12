"""Drop the bad real labels found by audit_labels.py (+ visual review of autobox/audit_sheets).

Dropped: every label whose audit verdict is "other" (the model sees a different sign there) or "none"
(no sign found there - mostly loose boxes or random plates), and all labels of DROP_CLASSES whose
labels were found to box the wrong sign throughout.
The .txt files in Merged are rewritten in place (same inode), so the hard-linked copies in
standard_japan change with them; a frame left without labels gets an empty .txt and is skipped by
make_train_split.py (an empty real .txt means "dropped", not "no sign"). The json status becomes
"uncertain" with reason "audit drop". Backups: autobox_backup_audit/ (jsons + old .txt contents).
usage: python apply_audit.py
"""
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent
DATA = ROOT / "classified Data japan" / "Merged"
AUTO = ROOT / "autobox"
BAK = ROOT / "autobox_backup_audit"
DROP_CLASSES = {"Traffic Distribution by Direction of Proceeding", "Road Closed to Pedestrians"}


def main():
    if BAK.exists():
        raise SystemExit(f"{BAK} exists - audit already applied")
    (BAK / "txt").mkdir(parents=True)
    for jf in AUTO.glob("*.json"):
        shutil.copy2(jf, BAK / jf.name)
    drop = defaultdict(list)                  # (folder, file) -> boxes to drop
    for r in csv.DictReader(open(AUTO / "audit.csv")):
        if r["verdict"] in ("other", "none") or r["folder"] in DROP_CLASSES:
            drop[(r["folder"], r["file"])].append(json.loads(r["box"]))
    per_class, emptied = defaultdict(int), 0
    by_folder = defaultdict(dict)
    for (folder, f), boxes in drop.items():
        by_folder[folder][f] = boxes
    for folder, files in by_folder.items():
        jf = AUTO / f"{folder}.json"
        res = json.loads(jf.read_text())
        for f, bad in files.items():
            r = res[f]
            keep = [b for b in r["boxes"] if not any(max(abs(a - c) for a, c in zip(b, d)) < 0.11 for d in bad)]
            txt = DATA / folder / (Path(f).stem + ".txt")
            (BAK / "txt" / f"{folder}__{txt.name}").write_text(txt.read_text() if txt.exists() else "")
            cid = int(txt.read_text().split()[0]) if txt.exists() and txt.read_text().strip() else None
            with Image.open(DATA / folder / f) as im:
                W, H = im.size
            with open(txt, "w") as fh:                       # same inode: hard links follow
                for x0, y0, x1, y1 in keep:
                    fh.write(f"{cid} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n")
            per_class[folder] += len(r["boxes"]) - len(keep)
            if keep:
                r["boxes"] = keep
            else:
                r.update(status="uncertain", reason="audit drop", boxes=[])
                emptied += 1
        jf.write_text(json.dumps(res, indent=1))
    print(f"dropped {sum(per_class.values())} labels, {emptied} frames left without a label")
    for k, v in sorted(per_class.items(), key=lambda x: -x[1]):
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    main()
