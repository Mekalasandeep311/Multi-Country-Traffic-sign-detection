"""Demote accepted boxes whose shape does not fit the class's sign (run after restore_crosscheck.py).

For compact signs (round / square / triangle / diamond; template aspect 0.75-1.33) a box more than
TOL x taller or wider than the template usually spans two stacked signs. Panels and guide boards
(other aspects, very variable in real life) are not checked. Writes autobox/shape_check.csv.
"""
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from autobox_real import DATA, OUT, ROOT
from autolabel import load_template_rgba
from fix_labels_jp import write_txt

TOL = 1.5
# panels / guide boards whose handbook picture happens to be square but real boards vary a lot
NOT_COMPACT = {"Destination & Direction (type 1)", "Destination & Direction (type 2)",
               "Destination & Direction (type 3)", "Destination & Distance", "Designates Lane for Destination",
               "Expressway Entrance", "Expressway Exit", "Emergency Telephone", "Emergency Parking",
               "Parking Area", "National Route Marker and Highway Signs", "Exclusive Traffic Lanes",
               "Rest Area", "Rest Area - Parking - Fuel Stand", "Time-restricted Parking Area",
               "Parking of Vehicles Bearing Elderly and Special-needs Driver Marks Allowed"}


def template_aspect(cid):
    a = []
    for p in (ROOT / "templates").glob(f"{cid + 1:03d}_*/*.png"):
        al = load_template_rgba(p)[..., 3] > 127
        ys, xs = np.nonzero(al)
        if len(xs):
            a.append((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1))
    return float(np.median(a)) if a else None


def undo():
    """Re-accept everything a previous shape_check run demoted (it only demotes accepted boxes)."""
    p = OUT / "shape_check.csv"
    if not p.exists():
        return
    folders = yaml.safe_load((ROOT / "class_map.yaml").read_text())["folders"]
    rows = list(csv.DictReader(open(p)))
    for folder in {r["class"] for r in rows}:
        jf = OUT / f"{folder}.json"
        res = json.loads(jf.read_text())
        for r in rows:
            if r["class"] != folder:
                continue
            x = res[r["file"]]
            if x["status"] == "uncertain" and x.get("reason", "").startswith("shape:"):
                x.update(status="accepted", reason="")
                write_txt(folder, r["file"], folders[folder], x["boxes"])
        jf.write_text(json.dumps(res, indent=1))
    p.unlink()
    print("undid", len(rows), "shape demotions")


def main():
    undo()
    folders = yaml.safe_load((ROOT / "class_map.yaml").read_text())["folders"]
    rows = []
    for folder, cid in folders.items():
        jf = OUT / f"{folder}.json"
        ta = template_aspect(cid)
        if not jf.exists() or ta is None or not 0.75 <= ta <= 1.33 or folder in NOT_COMPACT:
            continue
        res = json.loads(jf.read_text())
        n = 0
        for f, r in res.items():
            if r["status"] != "accepted" or not r["boxes"]:
                continue
            x0, y0, x1, y1 = r["boxes"][0]
            a = (x1 - x0) / max(1e-3, y1 - y0)
            if abs(np.log(a / ta)) > np.log(TOL):
                r.update(status="uncertain", reason=f"shape: box aspect {a:.2f} vs sign {ta:.2f}")
                (DATA / folder / (Path(f).stem + ".txt")).unlink(missing_ok=True)
                rows.append([folder, f, round(a, 3), round(ta, 3)])
                n += 1
        if n:
            jf.write_text(json.dumps(res, indent=1))
            print(f"{folder}: {n} demoted (sign aspect {ta:.2f})", flush=True)
    with open(OUT / "shape_check.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "file", "box_aspect", "sign_aspect"])
        w.writerows(rows)
    print("total demoted by shape:", len(rows))


if __name__ == "__main__":
    main()
