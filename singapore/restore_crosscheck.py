"""Undo cross-check demotions where the evidence for the folder's class is strong and YOLO's
"other class" vote is only moderate (typically look-alike pairs YOLO confuses, e.g.
Works Ahead / Work Access). Rewrites the .txt and the autobox/<class>.json status.

usage: python restore_crosscheck.py
"""
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).parent
DATA = ROOT / "Traffic Signs Singapore"
OUT = ROOT / "autobox"
OTHER_MAX = 0.60
VERIFIED_LOOKALIKES = {("Works Ahead", "Work Access")}  # checked visually: folder picks are correct


def main():
    folders = yaml.safe_load((ROOT / "class_map.yaml").read_text())["folders"]
    with open(OUT / "crosscheck.csv") as fh:
        demoted = list(csv.DictReader(fh))
    by_folder = {}
    for r in demoted:
        by_folder.setdefault(r["class"], []).append(r)
    restored = 0
    for folder, rows in by_folder.items():
        cid = folders[folder]
        jf = OUT / f"{folder}.json"
        res = json.loads(jf.read_text())
        scores = [r["score"] for r in res.values() if r.get("score") is not None and r["status"] in ("accepted", "uncertain")]
        med = float(np.median(scores)) if scores else 1.0
        n = 0
        for row in rows:
            r = res[row["file"]]
            if r["status"] == "uncertain" and r["score"] >= med and (float(row["other_score"]) < OTHER_MAX or (folder, row["yolo_other_class"]) in VERIFIED_LOOKALIKES):
                with Image.open(DATA / folder / row["file"]) as im:
                    W, H = im.size
                (DATA / folder / (Path(row["file"]).stem + ".txt")).write_text("".join(
                    f"{cid} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n"
                    for x0, y0, x1, y1 in r["boxes"]))
                r["status"], r["reason"] = "accepted", "restored: strong folder evidence"
                n += 1
        jf.write_text(json.dumps(res, indent=1))
        restored += n
        print(f"{folder}: restored {n}/{len(rows)}")
    print("total restored:", restored)


if __name__ == "__main__":
    main()
