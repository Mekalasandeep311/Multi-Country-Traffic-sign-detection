"""Japan-specific label fixes, run after `autobox_real.py select` and before crosscheck_yolo.py.

1. Plate classes. A supplementary plate hangs under a main sign, and the auto-boxer tends to pick the
   main sign. Re-pick per frame: among the cached OWLv2 candidates (+ the synthetic YOLO's own top
   boxes for the class), take a white plate-shaped box directly under a round main sign.
   Japanese plates often combine the time ("8-20") and the zone arrow on ONE board. The plate is split
   into its text band (dark, not red) and its arrow band (red): Hours Regulated boxes only the text
   part, the control-zone classes only the arrow part. Folder part not found -> uncertain (no .txt).
2. Shared frames. The same frame sits in several class folders (one per sign it shows). If two
   folders accepted the SAME object (IoU > 0.5) for different classes, the synthetic YOLO's class
   scores at that box decide; the other label is demoted to uncertain.

Writes autobox/plates.csv, autobox/shared_conflicts.csv; updates autobox/<folder>.json and the .txt
files beside the images.
"""
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "models"))

import cv2
import numpy as np
import torch
import torchvision
import yaml
from PIL import Image

from autobox_real import DATA, DEVICE, OUT, iou_matrix
from crosscheck_yolo import YoloAll

PLATES = {                                   # folder -> which part of the plate is this class
    "Hours Regulated": "text",
    "Start of control zone": "arrow",
    "Within control zone": "arrow",
    "End of control (applies to sign below)": "arrow",
}
MAIN_SIGN = 0.18          # OWL sign-ness of the main sign above the plate
WHITE_MIN = 0.35          # plates are white: share of low-saturation, not-dark pixels
RED_ARROW = 0.03          # share of red pixels: arrow plate >= this, text plate < this
SHARED_IOU = 0.5


def colour_shares(rgb, box):
    """(red share, white share) inside the box."""
    x0, y0, x1, y1 = map(int, box)
    c = rgb[max(0, y0):y1, max(0, x0):x1]
    if c.size == 0:
        return 0.0, 0.0
    hsv = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((h < 10) | (h > 170)) & (s > 90) & (v > 70)).mean()
    white = ((s < 60) & (v > 0.6 * np.percentile(v, 95))).mean()
    return float(red), float(white)


def runs(mask):
    """Contiguous True runs (start, end_exclusive), bridging 1-row gaps."""
    m = mask.copy()
    m[1:-1] |= mask[:-2] & mask[2:]
    out, start = [], None
    for i, v in enumerate(list(m) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i)); start = None
    return out


def split_plate(rgb, p):
    """-> {"text": box or None, "arrow": box or None}; bands keep the full plate width."""
    x0, y0, x1, y1 = map(int, p)
    c = rgb[max(0, y0):y1, max(0, x0):x1]
    if c.shape[0] < 6 or c.shape[1] < 6:
        return {"text": None, "arrow": None}
    hsv = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    red = ((h < 10) | (h > 170)) & (s > 90) & (v > 70)
    dark = (v < 0.5 * np.percentile(v, 95)) & (s < 110) & ~red
    red_rows = red.mean(1) > 0.06
    ar = max(runs(red_rows), key=lambda r: r[1] - r[0], default=None)
    if ar is None or ar[1] - ar[0] < 2:
        # no arrow: a pure text plate if it has dark ink
        return {"text": [x0, y0, x1, y1] if dark.mean() > 0.04 else None, "arrow": None}
    dark_rows = dark.mean(1) > 0.05
    dark_rows[max(0, ar[0] - 1):ar[1] + 1] = False
    tr = max(runs(dark_rows), key=lambda r: r[1] - r[0], default=None)
    if tr is None or tr[1] - tr[0] < 3:
        return {"text": None, "arrow": [x0, y0, x1, y1]}          # pure arrow plate
    if tr[1] <= ar[0]:                                           # text above arrow (usual)
        mid = (tr[1] + ar[0]) // 2
        return {"text": [x0, y0, x1, y0 + mid], "arrow": [x0, y0 + mid, x1, y1]}
    mid = (ar[1] + tr[0]) // 2                                   # arrow above text
    return {"text": [x0, y0 + mid, x1, y1], "arrow": [x0, y0, x1, y0 + mid]}


def write_txt(folder, f, cid, boxes):
    txt = DATA / folder / (Path(f).stem + ".txt")
    if not boxes:
        txt.unlink(missing_ok=True)
        return
    with Image.open(DATA / folder / f) as im:
        W, H = im.size
    txt.write_text("".join(f"{cid} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                           f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n" for x0, y0, x1, y1 in boxes))


def fix_plates(folders):
    rows = []
    for folder, part in PLATES.items():
        if folder not in folders or not (OUT / f"{folder}.json").exists():
            continue
        cid = folders[folder]
        res = json.loads((OUT / f"{folder}.json").read_text())
        z = np.load(OUT / "cache" / f"{folder}.npz")
        files, idx, boxes, sign = list(z["files"]), z["img_idx"], z["boxes"], z["sign"]
        y = np.load(OUT / "cache" / f"{folder}_yolo.npz")
        n_acc = 0
        for i, f in enumerate(files):
            f = str(f)
            c = boxes[idx == i]
            s = sign[idx == i]
            w, h = c[:, 2] - c[:, 0], c[:, 3] - c[:, 1]
            mains = c[(s >= MAIN_SIGN) & (w / np.maximum(h, 1) > 0.7) & (w / np.maximum(h, 1) < 1.4)]
            plates = c[(w / np.maximum(h, 1) > 1.3) & (w / np.maximum(h, 1) < 7) & (h >= 5)]
            yk = y["scores"][i] >= 0.05
            plates = np.concatenate([plates, y["boxes"][i][yk]]) if yk.any() else plates
            rgb = None
            best = None
            for m in mains:
                mw, mh = m[2] - m[0], m[3] - m[1]
                for p in plates:
                    pw, ph = p[2] - p[0], p[3] - p[1]
                    if not (m[3] - 0.15 * mh <= p[1] <= m[3] + 1.3 * mh and
                            abs((p[0] + p[2]) / 2 - (m[0] + m[2]) / 2) < 0.35 * mw and
                            0.6 * mw <= pw <= 1.9 * mw and 0.12 * mw <= ph <= 0.75 * mw):
                        continue
                    if rgb is None:
                        rgb = np.array(Image.open(DATA / folder / f).convert("RGB"))
                    red, white = colour_shares(rgb, p)
                    if white < WHITE_MIN:
                        continue
                    box = split_plate(rgb, p)[part]
                    if box is None:
                        continue
                    ov = iou_matrix(np.array([box], float), y["boxes"][i])[0]
                    ys = float(np.where(ov >= 0.3, y["scores"][i], 0).max())
                    key = ys - 0.002 * (p[1] - m[3]) / max(mh, 1)          # prefer the plate closest to the sign
                    if best is None or key > best[0]:
                        best = (key, np.array(box, float), ys, red)
            r = res[f]
            if best is None:
                r.update(status="uncertain", reason="plate: no plate under a sign", boxes=[])
                write_txt(folder, f, cid, [])
            else:
                _, p, ys, red = best
                r.update(status="accepted", reason="plate under sign", score=round(ys, 4),
                         boxes=[[round(float(v), 1) for v in p]])
                write_txt(folder, f, cid, r["boxes"])
                n_acc += 1
            rows.append([folder, f, r["status"], r.get("score"), round(best[3], 3) if best else ""])
        (OUT / f"{folder}.json").write_text(json.dumps(res, indent=1))
        print(f"plates {folder}: {n_acc}/{len(files)} accepted", flush=True)
    with open(OUT / "plates.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "file", "status", "yolo_score", "red_share"])
        w.writerows(rows)


def resolve_shared(folders):
    results = {f: json.loads((OUT / f"{f}.json").read_text()) for f in folders if (OUT / f"{f}.json").exists()}
    by_frame = defaultdict(list)                      # (name, size) -> [(folder, file)]
    for folder, res in results.items():
        for f, r in res.items():
            if r["status"] == "accepted" and r["boxes"]:
                by_frame[(f, (DATA / folder / f).stat().st_size)].append((folder, f))
    yolo, rows, demoted = None, [], set()
    for key, items in by_frame.items():
        if len({folders[fo] for fo, _ in items}) < 2:
            continue
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                (fa, na), (fb, nb) = items[a], items[b]
                if folders[fa] == folders[fb] or (fa, na) in demoted or (fb, nb) in demoted:
                    continue
                ba, bb = results[fa][na]["boxes"][0], results[fb][nb]["boxes"][0]
                if iou_matrix([ba], [bb])[0, 0] <= SHARED_IOU:
                    continue
                yolo = yolo or YoloAll()
                boxes, cls = yolo(np.array(Image.open(DATA / fa / na).convert("RGB")))
                m = torchvision.ops.box_iou(torch.tensor([ba], device=DEVICE, dtype=torch.float32), boxes)[0] >= 0.5
                sa = float(cls[m][:, folders[fa]].max()) if m.any() else 0.0
                sb = float(cls[m][:, folders[fb]].max()) if m.any() else 0.0
                lose = (fb, nb) if sa >= sb else (fa, na)
                win = (fa, na) if lose == (fb, nb) else (fb, nb)
                r = results[lose[0]][lose[1]]
                r.update(status="uncertain", reason=f"shared frame: same box labeled {win[0]} in that folder")
                write_txt(lose[0], lose[1], folders[lose[0]], [])
                demoted.add(lose)
                rows.append([key[0], fa, round(sa, 3), fb, round(sb, 3), lose[0]])
    for folder in {fo for fo, _ in demoted}:
        (OUT / f"{folder}.json").write_text(json.dumps(results[folder], indent=1))
    with open(OUT / "shared_conflicts.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "folder_a", "score_a", "folder_b", "score_b", "demoted"])
        w.writerows(rows)
    print(f"shared-frame conflicts: {len(rows)} labels demoted", flush=True)
    for fo in sorted({r[5] for r in rows}):
        print(f"  {fo}: {sum(r[5] == fo for r in rows)}")


def main():
    folders = yaml.safe_load((Path(__file__).parent / "class_map.yaml").read_text())["folders"]
    fix_plates(folders)
    resolve_shared(folders)


if __name__ == "__main__":
    main()
