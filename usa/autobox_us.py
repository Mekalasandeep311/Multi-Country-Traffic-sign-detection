"""Auto-box EVERY sign in the US real frames and name its class (there are no hand-sorted folders).

For each kept frame of real/frames.csv:
  A. candidates  OWLv2 on full frame + 6 tiles (autobox_real.Owl): box, sign-ness, box embedding.
  B. two votes   - the synthetic-trained YOLO (runs/us_synth): per-class scores of the anchors
                   overlapping the box (IoU >= 0.5); its best class c and score p.
                 - OWLv2 one-shot template match (autobox_real.ClassBank): c must be among the
                   OWL_TOPK closest handbook templates of the box.
  C. decide      accepted   p >= YOLO_ACCEPT and both votes agree
                 uncertain  sign-like box that is not accepted (no label; greyed out in training tiles)
                 Accepted boxes of different classes overlapping by IoU >= 0.5: the higher p wins.
Phase "frames" writes autobox/frames.jsonl (resumable). Phase "write" builds, from it:
  real_by_class/<class>/<source>__<frame>.jpg + .txt   hard link per class present in the frame;
                                                       .txt = ALL accepted signs of the frame, the
                                                       folder's own class first
  autobox/<class>.json, autobox/cache/<class>.npz      same format as Japan (build_balanced_dataset.py)
  autobox/uncertain.json                               {frame: [uncertain boxes]} for training
  autobox/review/<class>.jpg, autobox/spot_check.jpg   contact sheets; autobox/report.csv

usage: python autobox_us.py [frames|redecide|write|all]   (redecide also runs write)
"""
import csv
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "models"))

import cv2
import numpy as np
import torch
import torchvision
import yaml
from PIL import Image

from autobox_real import DATA, DEVICE, OUT, ClassBank, Owl, crop_of
from crosscheck_yolo import YoloAll

ROOT = Path(__file__).parent
SIGN_GATE = 0.12        # OWLv2 sign-ness a candidate needs
OTHER_SIGN = 0.20       # sign-ness kept as "other sign" for donor inpainting
YOLO_ACCEPT = 0.30
YOLO_UNCERTAIN = 0.10
OWL_TOPK = 5
YOLO_ONLY = 0.50        # YOLO boxes OWLv2 missed: kept as uncertain from this score
JSONL = OUT / "frames.jsonl"
JSONL2 = OUT / "frames_v2.jsonl"   # after redecide
YOLO_STRONG = 0.65      # redecide: YOLO class alone is enough from this score
MIN_SIDE = 18           # px; smaller (blurry) boxes stay uncertain
DARK_V, DARK_FRAC = 60, 0.35
# never auto-accepted (look-alikes YOLO cannot tell apart from appearance): Stop PARE, guide signs,
# toll plazas, street-name signs, route shields
NO_AUTO = {1} | set(range(167, 209)) | {90, 91, 112, 128, 132, 166}   # + yellow text-only diamonds (DIP, ROUGH ROAD... look the same)
# + white/red text plates and speed-limit / parking variants: real US streets carry many similar plates
# that are not among the 209 classes (NO TURNS, ALL TRAFFIC, STOP HERE ON RED ...), and YOLO maps them
# onto these classes even at p >= 0.9; also two pictogram classes that only caught look-alikes
NO_AUTO |= ({4, 10, 11, 13, 14, 15, 20, 23, 24, 25, 26, 35, 38} | set(range(39, 67)) | {68}
            | set(range(69, 74)) | {75} | set(range(77, 84)) | set(range(86, 90)) | {145, 155})
# need a very confident YOLO score
STRICT = set(range(5, 9)) | {12, 19, 116, 117} | set(range(157, 163))
STRICT_P = 0.90
MIN_FRAMES = 5          # a class found in fewer frames is mostly look-alikes: kept as uncertain


def safe(name):          # same as build_balanced_dataset.safe (that module reads sys.argv on import)
    return name.replace("/", "-").replace(":", "-").strip()


def frames_phase():
    with open(ROOT / "real" / "frames.csv") as fh:
        paths = [r["path"] for r in csv.DictReader(fh) if r["kept"] == "1"]
    done = set()
    if JSONL.exists():
        done = {json.loads(l)["path"] for l in open(JSONL)}
    todo = [p for p in paths if p not in done]
    print(f"{len(paths)} frames, {len(done)} done, {len(todo)} to do", flush=True)
    names = yaml.safe_load((ROOT / "class_map.yaml").read_text())["names"]
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    owl = Owl()
    bank = ClassBank(owl, names)
    yolo = YoloAll()
    t0 = time.time()
    with open(JSONL, "a") as out:
        for n, rel in enumerate(todo):
            rgb = np.array(Image.open(ROOT / rel).convert("RGB"))
            B, S, E = owl.candidates(rgb)
            keep = S >= SIGN_GATE
            B, S, E = B[keep], S[keep], E[keep]
            yb, ycls = yolo(rgb)
            acc, unc = [], []
            if len(B):
                p_img, _, _ = bank.probs(E)
                ov = torchvision.ops.box_iou(torch.tensor(B, dtype=torch.float32, device=DEVICE), yb)
                for j in range(len(B)):
                    m = ov[j] >= 0.5
                    if m.any():
                        s = ycls[m].max(0).values
                        c, p = int(s.argmax()), float(s.max())
                    else:
                        c, p = -1, 0.0
                    rank = int((p_img[j] > p_img[j, c]).sum()) if c >= 0 else 999
                    box = [round(float(v), 1) for v in B[j]]
                    if p >= YOLO_ACCEPT and rank < OWL_TOPK:
                        acc.append(box + [c, round(p, 3), rank])
                    elif p >= YOLO_UNCERTAIN or S[j] >= OTHER_SIGN:
                        unc.append(box + [c, round(p, 3), round(float(S[j]), 3)])
            # YOLO detections OWLv2 did not propose
            conf, cls = ycls.max(1)
            sel = conf >= YOLO_ONLY
            if sel.any():
                k = torchvision.ops.nms(yb[sel], conf[sel], 0.5)
                for bb, cc, pp in zip(yb[sel][k].tolist(), cls[sel][k].tolist(), conf[sel][k].tolist()):
                    near = [b[:4] for b in acc + unc]
                    if not near or torchvision.ops.box_iou(torch.tensor([bb]), torch.tensor(near)).max() < 0.3:
                        unc.append([round(v, 1) for v in bb] + [int(cc), round(pp, 3), 0.0])
            # one class per place: overlapping accepted boxes keep the higher YOLO score
            acc.sort(key=lambda a: -a[5])
            final = []
            for a in acc:
                if all(torchvision.ops.box_iou(torch.tensor([a[:4]]), torch.tensor([f[:4]]))[0, 0] < 0.5 for f in final):
                    final.append(a)
                else:
                    unc.append(a[:6] + [0.0])
            out.write(json.dumps({"path": rel, "size": [rgb.shape[1], rgb.shape[0]], "acc": final, "unc": unc,
                                  "other": [[round(float(v), 1) for v in b] for b, s in zip(B, S) if s >= OTHER_SIGN]}) + "\n")
            if n % 200 == 0:
                out.flush()
                print(f"{n}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)


def is_lamp(rgb, box):
    """Traffic-light lens: the box sits in a dark housing (many near-black pixels around a bright blob)."""
    x0, y0, x1, y1 = [int(round(v)) for v in box[:4]]
    c = rgb[max(0, y0):y1, max(0, x0):x1]
    if c.size == 0:
        return True
    v = c.max(2)
    return (v < DARK_V).mean() > DARK_FRAC


def redecide_phase():
    """Re-decide the saved boxes (no GPU): the OWLv2 template vote proved unreliable on real frames
    (it favours green rectangles), so a sign-like OWLv2 box is accepted on the YOLO class alone when
    p >= YOLO_STRONG. Look-alike classes (guide signs, street names, route shields, PARE) are never
    auto-accepted, and traffic-light lenses are dropped."""
    recs = []
    n_lamp = 0
    for line in open(JSONL):
        r = json.loads(line)
        cand, unc = [], []
        for a in r["acc"]:                      # old rule: YOLO >= 0.30 and OWL rank < 5
            cand.append(a[:6])
        for u in r["unc"]:
            if u[5] >= YOLO_STRONG and u[6] > 0:  # OWLv2 candidate with a strong YOLO class
                cand.append(u[:6])
            else:
                unc.append(u)
        keep = []
        rgb = None
        for a in cand:
            if (a[4] in NO_AUTO or (a[4] in STRICT and a[5] < STRICT_P)
                    or min(a[2] - a[0], a[3] - a[1]) < MIN_SIDE):
                unc.append(a + [0.0])
                continue
            if rgb is None:
                rgb = np.array(Image.open(ROOT / r["path"]).convert("RGB"))
            if is_lamp(rgb, a):
                n_lamp += 1
                continue                         # not a sign: no label, not greyed out
            keep.append(a)
        keep.sort(key=lambda a: -a[5])
        final = []
        for a in keep:
            if all(torchvision.ops.box_iou(torch.tensor([a[:4]]), torch.tensor([f[:4]]))[0, 0] < 0.5 for f in final):
                final.append(a)
            else:
                unc.append(a + [0.0])
        r["acc"], r["unc"] = final, unc
        recs.append(r)
    frames = defaultdict(int)
    for r in recs:
        for cid in {a[4] for a in r["acc"]}:
            frames[cid] += 1
    rare = {cid for cid, k in frames.items() if k < MIN_FRAMES}
    n_acc = 0
    with open(JSONL2, "w") as out:
        for r in recs:
            r["unc"] += [a + [0.0] for a in r["acc"] if a[4] in rare]
            r["acc"] = [a for a in r["acc"] if a[4] not in rare]
            n_acc += len(r["acc"])
            out.write(json.dumps(r) + "\n")
    print(f"redecide: {n_acc} accepted boxes, {n_lamp} traffic-light boxes dropped, {len(rare)} classes "
          f"under {MIN_FRAMES} frames demoted -> {JSONL2}", flush=True)


def write_phase():
    names = yaml.safe_load((ROOT / "class_map.yaml").read_text())["names"]
    recs = [json.loads(l) for l in open(JSONL2 if JSONL2.exists() else JSONL)]
    if DATA.exists():
        shutil.rmtree(DATA)          # only our own output (hard links + .txt) lives here
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    per_class = defaultdict(dict)    # cid -> {file: result}
    per_class_cache = defaultdict(lambda: {"files": [], "img_idx": [], "boxes": [], "sign": []})
    uncertain = {}
    for r in recs:
        if r["unc"]:
            uncertain[r["path"]] = [u[:4] for u in r["unc"]]
        if not r["acc"]:
            continue
        W, H = r["size"]
        src = ROOT / r["path"]
        fname = f"{src.parent.name}__{src.name}"
        for cid in sorted({a[4] for a in r["acc"]}):
            own = [a for a in r["acc"] if a[4] == cid]
            rest = [a for a in r["acc"] if a[4] != cid]
            d = DATA / safe(names[cid])
            d.mkdir(parents=True, exist_ok=True)
            os.link(src, d / fname)
            (d / fname).with_suffix(".txt").write_text("".join(
                f"{a[4]} {(a[0] + a[2]) / 2 / W:.6f} {(a[1] + a[3]) / 2 / H:.6f} {(a[2] - a[0]) / W:.6f} "
                f"{(a[3] - a[1]) / H:.6f}\n" for a in own + rest))
            per_class[cid][fname] = {"status": "accepted", "score": max(a[5] for a in own),
                                     "boxes": [a[:4] for a in own], "others": [a[:4] for a in rest]}
            c = per_class_cache[cid]
            i = len(c["files"])
            c["files"].append(fname)
            others = r["other"] + [u[:4] for u in r["unc"]] + [a[:4] for a in rest]
            c["img_idx"] += [i] * len(others)
            c["boxes"] += others
            c["sign"] += [1.0] * len(others)
    for cid, res in per_class.items():
        key = safe(names[cid])
        (OUT / f"{key}.json").write_text(json.dumps(res, indent=1))
        c = per_class_cache[cid]
        np.savez_compressed(OUT / "cache" / f"{key}.npz", files=np.array(c["files"]),
                            img_idx=np.array(c["img_idx"], int), boxes=np.array(c["boxes"], np.float32).reshape(-1, 4),
                            sign=np.array(c["sign"], np.float32))
    (OUT / "uncertain.json").write_text(json.dumps(uncertain))

    # report + review sheets
    (OUT / "review").mkdir(exist_ok=True)
    rows = []
    for cid in range(len(names)):
        res = per_class.get(cid, {})
        rows.append([cid, names[cid], len(res), sum(len(r["boxes"]) for r in res.values())])
        if not res:
            continue
        items = sorted(res.items(), key=lambda kv: kv[1]["score"])
        if len(items) > 60:
            items = items[:30] + [items[k] for k in np.linspace(30, len(items) - 1, 30).astype(int)]
        tiles = []
        for f, r in items:
            im = cv2.imread(str(DATA / safe(names[cid]) / f))
            c = cv2.resize(crop_of(im, r["boxes"][0], pad=0.6), (128, 128))
            cv2.putText(c, f"{r['score']:.2f}", (3, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            tiles.append(c)
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 10)
        cv2.imwrite(str(OUT / "review" / f"{safe(names[cid])}.jpg"),
                    np.vstack([np.hstack(tiles[k:k + 10]) for k in range(0, len(tiles), 10)]))
    # spot check: 100 random accepted boxes over all classes, class name printed
    allb = [(r["path"], a) for r in recs for a in r["acc"]]
    tiles = []
    for path, a in random.Random(0).sample(allb, min(100, len(allb))):
        im = cv2.imread(str(ROOT / path))
        x0, y0, x1, y1 = map(int, a[:4])
        m = int(0.8 * max(x1 - x0, y1 - y0)) + 8
        ox, oy = max(0, x0 - m), max(0, y0 - m)
        c = im[oy:y1 + m, ox:x1 + m].copy()
        cv2.rectangle(c, (x0 - ox, y0 - oy), (x1 - ox, y1 - oy), (0, 255, 0), 1)
        c = cv2.resize(c, (180, 180))
        cv2.rectangle(c, (0, 0), (180, 26), (0, 0, 0), -1)
        n = f"{a[4] + 1} {names[a[4]]}"
        cv2.putText(c, n[:30], (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
        cv2.putText(c, f"{n[30:58]}  {a[5]:.2f}", (2, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
        tiles.append(c)
    if tiles:
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 10)
        cv2.imwrite(str(OUT / "spot_check.jpg"), np.vstack([np.hstack(tiles[k:k + 10]) for k in range(0, len(tiles), 10)]))
    with open(OUT / "report.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class_id", "class", "frames", "boxes"])
        w.writerows(rows)
    n_acc = sum(len(r["acc"]) for r in recs)
    n_unc = sum(len(r["unc"]) for r in recs)
    print(f"{len(recs)} frames: {sum(1 for r in recs if r['acc'])} with accepted signs, {n_acc} accepted boxes, "
          f"{n_unc} uncertain; {sum(1 for r in rows if r[2])} of {len(names)} classes have real frames")


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    if phase in ("frames", "all"):
        frames_phase()
    if phase in ("redecide", "all"):
        redecide_phase()
    if phase in ("write", "redecide", "all"):
        write_phase()
