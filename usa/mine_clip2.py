"""Hand-checked real training frames from infer_video/clip2.mp4 (user request 2026-11-02: train the frames the
v3 video test got wrong, incl. the missed NYC bus-stop sign).

Phases
  props   sample clip2 every STEP frames; per frame: OWLv2 sign-like boxes (sign-ness >= OWL_MIN) + v3 boxes
          (conf >= V3_MIN), each OWL box tagged with the best overlapping v3 class -> clip2_mine/props.jsonl,
          frames -> clip2_mine/frames/
  tracks  link boxes over consecutive sampled frames into sign tracks, keep plausible ones, write review sheets
          (one crop per track: its largest box, with v3's guess) -> clip2_mine/tracks.json, sheets/
  write   read clip2_mine/decisions.txt ("<track> <class id>" | "<track> o" other sign | "<track> g" grey |
          "<track> n" not a sign) and write labelled frames: clip2_labeled/<frame>.jpg + .txt (uncertain and
          undecided sign tracks greyed, labels never greyed)
usage: python mine_clip2.py props|tracks|write
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
VIDEO = ROOT / "infer_video" / "clip2.mp4"
WEIGHTS = ROOT / "runs" / "us_signs_v3" / "weights" / "best.pt"
OUT = ROOT / "clip2_mine"
LABELED = ROOT / "clip2_labeled"
STEP = 12                 # 60 fps -> 5 sampled frames per second
OWL_MIN = 0.15
V3_MIN = 0.10
OTHER_ID = 209
GREY = 114


def iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def inside(a, b):
    """share of box a's area that lies inside box b"""
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    return 0.0 if iw <= 0 or ih <= 0 else iw * ih / max(1.0, (a[2] - a[0]) * (a[3] - a[1]))


def merge(owl_raw, v3_raw):
    """One tight box per sign: v3 boxes first (tight, with class); OWLv2 boxes only where v3 has none, and never
    a group box around several signs or a part of a sign. Box = [x0, y0, x1, y1, owl, v3 class, v3 conf]."""
    out = []
    for q in sorted((q for q in v3_raw if q[5] >= 0.15), key=lambda q: -q[5]):   # NMS across classes
        if all(iou(q, o) < 0.5 for o in out):
            s = max((o[4] for o in owl_raw if iou(q, o) >= 0.4), default=0.0)
            out.append(q[:4] + [s, q[4], q[5]])
    owl = [o for o in owl_raw if all(iou(o, q) < 0.3 for q in out)]
    for o in sorted(owl, key=lambda o: -(o[2] - o[0]) * (o[3] - o[1])):         # large first: parts drop later
        area = (o[2] - o[0]) * (o[3] - o[1])
        inner = [x for x in out + owl if x is not o and inside(x, o) >= 0.8
                 and area >= 1.8 * (x[2] - x[0]) * (x[3] - x[1])]
        group = any(iou(a, b) < 0.3 for i, a in enumerate(inner) for b in inner[i + 1:])  # >= 2 separate signs
        part = any(inside(o, x) >= 0.8 for x in out)
        if not group and not part:
            out.append(o[:4] + [o[4], -1, 0.0])
    return out


def props():
    from ultralytics import YOLO
    from autobox_real import Owl
    owl, m = Owl(), YOLO(str(WEIGHTS))
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with open(OUT / "props.jsonl", "w") as fh:
        for idx in range(0, n, STEP):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok:
                break
            cv2.imwrite(str(OUT / "frames" / f"f{idx:06d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            B, S, _ = owl.candidates(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            r = m.predict(bgr, imgsz=1280, conf=V3_MIN, verbose=False)[0]
            yb = r.boxes.xyxy.tolist(); yc = r.boxes.conf.tolist(); yk = [int(k) for k in r.boxes.cls.tolist()]
            owl_raw = [[round(v, 1) for v in b] + [round(s, 3)]
                       for b, s in zip(np.asarray(B).tolist(), np.asarray(S).tolist()) if s >= OWL_MIN]
            v3_raw = [[round(v, 1) for v in q] + [k, round(c, 3)] for q, c, k in zip(yb, yc, yk)]
            fh.write(json.dumps({"frame": idx, "boxes": merge(owl_raw, v3_raw), "owl": owl_raw, "v3": v3_raw}) + "\n")
            if idx % 1200 == 0:
                print(f"{idx}/{n}", flush=True)


def tracks():
    import yaml
    names = yaml.safe_load((ROOT / "us_train_v3" / "data.yaml").read_text())["names"]
    recs = [json.loads(l) for l in open(OUT / "props.jsonl")]
    active, done, tid = [], [], 0
    for r in recs:
        f = r["frame"]
        new_active, used = [], set()
        for t in active:
            last = t["boxes"][-1]
            lb = last[1]
            cands = []
            for j, b in enumerate(r["boxes"]):
                if j in used:
                    continue
                w0, h0 = lb[2] - lb[0], lb[3] - lb[1]
                w1, h1 = b[2] - b[0], b[3] - b[1]
                ratio = (w1 * h1) / max(1.0, w0 * h0)
                d = np.hypot((b[0] + b[2] - lb[0] - lb[2]) / 2, (b[1] + b[3] - lb[1] - lb[3]) / 2)
                if 0.4 < ratio < 2.5 and (iou(lb, b) > 0.2 or d < 1.5 * max(w0, h0, 8)):
                    cands.append((iou(lb, b) - d / 1000, j))
            if cands:
                _, j = max(cands)
                used.add(j)
                t["boxes"].append((f, r["boxes"][j]))
                new_active.append(t)
            elif f - last[0] <= 2 * STEP:          # allow one missed sample
                new_active.append(t)
            else:
                done.append(t)
        for j, b in enumerate(r["boxes"]):
            if j not in used:
                new_active.append({"id": tid, "boxes": [(f, b)]}); tid += 1
        active = new_active
    done += active
    keep = []
    for t in done:
        bs = [b for _, b in t["boxes"]]
        s_max = max(b[4] for b in bs)
        votes = defaultdict(float)
        for b in bs:
            if b[5] >= 0:
                votes[b[5]] += b[6]
        v3 = max(votes, key=votes.get) if votes else -1
        c_max = max((b[6] for b in bs if b[5] == v3), default=0.0)
        big = max(bs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        if len(bs) >= 2 and (s_max >= 0.25 or c_max >= 0.25) and min(big[2] - big[0], big[3] - big[1]) >= 10:
            t.update(v3=v3, v3_conf=round(c_max, 3), owl=round(s_max, 3), n=len(bs))
            keep.append(t)
    keep.sort(key=lambda t: (t["v3"] if t["v3"] >= 0 else 999, -t["v3_conf"]))
    (OUT / "tracks.json").write_text(json.dumps(keep))
    print(f"{len(done)} tracks, {len(keep)} kept for review")
    sheets = OUT / "sheets"
    sheets.mkdir(exist_ok=True)
    for old in sheets.glob("*.jpg"):
        old.unlink()
    tiles = []
    for t in keep:
        f, b = max(t["boxes"], key=lambda fb: (fb[1][2] - fb[1][0]) * (fb[1][3] - fb[1][1]))
        im = cv2.imread(str(OUT / "frames" / f"f{f:06d}.jpg"))
        x0, y0, x1, y1 = map(int, b[:4])
        m = int(1.0 * max(x1 - x0, y1 - y0)) + 12
        ox, oy = max(0, x0 - m), max(0, y0 - m)
        cr = im[oy:y1 + m, ox:x1 + m].copy()
        cv2.rectangle(cr, (x0 - ox, y0 - oy), (x1 - ox, y1 - oy), (0, 255, 0), 1)
        cr = cv2.resize(cr, (240, 240))
        cv2.rectangle(cr, (0, 0), (240, 32), (0, 0, 0), -1)
        g = names[t["v3"]][:30] if t["v3"] >= 0 else "-"
        cv2.putText(cr, f"#{t['id']}  n={t['n']}", (2, 13), 0, 0.42, (0, 255, 255), 1)
        cv2.putText(cr, f"{t['v3']}:{g} {t['v3_conf']:.2f}", (2, 28), 0, 0.36, (255, 255, 255), 1)
        tiles.append(cr)
    for s in range(0, len(tiles), 48):
        part = tiles[s:s + 48]
        part += [np.zeros_like(tiles[0])] * (-len(part) % 8)
        cv2.imwrite(str(sheets / f"sheet_{s // 48:02d}.jpg"),
                    np.vstack([np.hstack(part[i:i + 8]) for i in range(0, len(part), 8)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{(len(tiles) + 47) // 48} sheets")


def write():
    trk = {t["id"]: t for t in json.loads((OUT / "tracks.json").read_text())}
    dec = {}
    for line in (OUT / "decisions.txt").read_text().split("\n"):
        line = line.split("#")[0].strip()
        if line:
            a, b = line.split()
            dec[int(a)] = b
    per_frame = defaultdict(lambda: {"lab": [], "grey": []})
    for i, t in trk.items():
        # unreviewed track: grey if it looks like a sign (OWLv2 >= 0.4), else background (shop fronts, windows)
        d = dec.get(i, "g" if t["owl"] >= 0.4 else "n")
        for f, b in t["boxes"]:
            if d == "n":
                continue
            if d == "g":
                per_frame[f]["grey"].append(b[:4])
            else:
                per_frame[f]["lab"].append((OTHER_ID if d == "o" else int(d), b[:4]))
    LABELED.mkdir(exist_ok=True)
    for old in LABELED.glob("*"):
        old.unlink()
    n_img = n_box = 0
    for f, v in sorted(per_frame.items()):
        if not v["lab"]:
            continue
        im = cv2.imread(str(OUT / "frames" / f"f{f:06d}.jpg"))
        H, W = im.shape[:2]
        mask = np.zeros((H, W), bool)
        for g in v["grey"]:
            mask[max(0, int(g[1])):int(np.ceil(g[3])), max(0, int(g[0])):int(np.ceil(g[2]))] = True
        for _, b in v["lab"]:
            mask[max(0, int(b[1])):int(np.ceil(b[3])), max(0, int(b[0])):int(np.ceil(b[2]))] = False
        im[mask] = GREY
        stem = f"clip2__f{f:06d}"
        cv2.imwrite(str(LABELED / f"{stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (LABELED / f"{stem}.txt").write_text("".join(
            f"{c} {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}\n"
            for c, b in v["lab"]))
        n_img += 1; n_box += len(v["lab"])
    print(f"labelled frames {n_img}, boxes {n_box}, decided tracks {len(dec)} of {len(trk)}")


if __name__ == "__main__":
    {"props": props, "tracks": tracks, "write": write}[sys.argv[1]]()
