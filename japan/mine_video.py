"""Hard-mining from a dashcam video (detections from detect.py).

events: group the detections into sign passes (same class, gaps <= GAP seconds) and draw one review
        crop per pass (the most confident box) with its event id -> events.csv + sheet_*.jpg
tiles:  after review (verdicts.csv: event_id,verdict with verdict ok / wrong / skip / cls:<id>) cut 640
        tiles from every STEP-th frame of each pass:
          ok    -> tile around the box, labeled with every ok box inside the tile (real sign examples)
          cls:N -> a real sign detected with the wrong class: as ok, but labeled class N
          wrong -> tile around the false detection, labeled only with ok boxes inside (hard negatives)
          skip  -> unsure: not used, and any tile that cuts into a skipped box is dropped (it could be
                   an unlabeled real sign)
        -> <out>/*.jpg + *.txt

usage: python mine_video.py events <detections.csv> <video> <out_dir>
       python mine_video.py tiles  <detections.csv> <video> <out_dir>   (reads <out_dir>/verdicts.csv)
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

GAP = 1.0
STEP = 10
TILE = 640


def load_events(det_csv):
    rows = sorted(csv.DictReader(open(det_csv)), key=lambda r: (r["class_name"], int(r["frame"])))
    events, cur = [], {}
    for r in rows:
        k, t = r["class_name"], float(r["time_s"])
        e = cur.get(k)
        if e is None or t - e["t_last"] > GAP:
            e = {"id": len(events), "class_id": int(r["class_id"]), "class": k, "rows": []}
            events.append(e)
            cur[k] = e
        e["rows"].append(r)
        e["t_last"] = t
    return events


def box(r):
    return [float(r[c]) for c in ("x0", "y0", "x1", "y1")]


def events_mode(det_csv, video, out):
    out.mkdir(parents=True, exist_ok=True)
    events = load_events(det_csv)
    cap = cv2.VideoCapture(str(video))
    tiles = []
    with open(out / "events.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["event_id", "class", "t_start", "t_end", "n_boxes", "max_conf"])
        for e in events:
            best = max(e["rows"], key=lambda r: float(r["conf"]))
            w.writerow([e["id"], e["class"], e["rows"][0]["time_s"], e["rows"][-1]["time_s"], len(e["rows"]), best["conf"]])
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(best["frame"]))
            ok, im = cap.read()
            x0, y0, x1, y1 = box(best)
            m = 1.2 * max(x1 - x0, y1 - y0) + 10
            ox, oy = int(max(0, x0 - m)), int(max(0, y0 - m))
            cr = im[oy:int(y1 + m), ox:int(x1 + m)].copy()
            cv2.rectangle(cr, (int(x0) - ox, int(y0) - oy), (int(x1) - ox, int(y1) - oy), (0, 255, 0), 1)
            cr = cv2.resize(cr, (200, 200))
            cv2.rectangle(cr, (0, 0), (200, 30), (0, 0, 0), -1)
            cv2.putText(cr, f"#{e['id']} {e['class'][:26]}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
            cv2.putText(cr, f"{float(best['conf']):.2f} n={len(e['rows'])} t={float(best['time_s']):.0f}s", (2, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1)
            tiles.append(cr)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 8)
    strips = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
    for j in range(0, len(strips), 5):
        cv2.imwrite(str(out / f"sheet_{j // 5:02d}.jpg"), np.vstack(strips[j:j + 5]))
    print(f"{len(events)} events, {(len(strips) + 4) // 5} sheets")


def tiles_mode(det_csv, video, out):
    verdict = {int(r["event_id"]): r["verdict"].strip() for r in csv.DictReader(open(out / "verdicts.csv"))}
    events = load_events(det_csv)
    ok_by_frame, skip_by_frame, jobs = defaultdict(list), defaultdict(list), defaultdict(list)
    for e in events:
        v = verdict.get(e["id"], "skip")
        if v.startswith("cls:"):
            e["class_id"], v = int(v[4:]), "ok"
        for r in e["rows"]:
            if v == "ok":
                ok_by_frame[int(r["frame"])].append((e["class_id"], box(r)))
            elif v != "wrong":
                skip_by_frame[int(r["frame"])].append(box(r))
        if v in ("ok", "wrong"):
            for r in e["rows"][::STEP]:
                jobs[int(r["frame"])].append((v, box(r), e["id"]))
    tile_dir = out / "tiles"
    tile_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    rng = np.random.default_rng(0)
    n = {"ok": 0, "wrong": 0, "dropped": 0}
    for f in sorted(jobs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if not ok:
            continue
        H, W = im.shape[:2]
        for kind, (x0, y0, x1, y1), eid in jobs[f]:
            tx = int(np.clip(rng.uniform(x1 - TILE, x0), 0, max(0, W - TILE)))
            ty = int(np.clip(rng.uniform(y1 - TILE, y0), 0, max(0, H - TILE)))
            tile = im[ty:ty + TILE, tx:tx + TILE]
            th, tw = tile.shape[:2]
            if any(min(a1, tx + tw) - max(a0, tx) > 0.3 * (a1 - a0) and min(b1, ty + th) - max(b0, ty) > 0.3 * (b1 - b0)
                   for a0, b0, a1, b1 in skip_by_frame.get(f, [])):
                n["dropped"] += 1
                continue
            lines = []
            for c, (a0, b0, a1, b1) in ok_by_frame.get(f, []):
                a0, a1 = np.clip([a0 - tx, a1 - tx], 0, tw)
                b0, b1 = np.clip([b0 - ty, b1 - ty], 0, th)
                if a1 - a0 > 2 and b1 - b0 > 2:
                    lines.append(f"{c} {(a0 + a1) / 2 / tw:.6f} {(b0 + b1) / 2 / th:.6f} {(a1 - a0) / tw:.6f} {(b1 - b0) / th:.6f}")
            stem = tile_dir / f"{kind}_e{eid:04d}_f{f:06d}"
            cv2.imwrite(f"{stem}.jpg", tile)
            Path(f"{stem}.txt").write_text("".join(l + "\n" for l in lines))
            n[kind] += 1
    print(f"tiles: {n['ok']} sign tiles, {n['wrong']} hard-negative tiles, {n['dropped']} dropped "
          f"(contain an unsure box) -> {tile_dir}")


if __name__ == "__main__":
    mode, det_csv, video, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    (events_mode if mode == "events" else tiles_mode)(det_csv, video, out)
