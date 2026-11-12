"""Review sheets for clip2_mine tracks: A = tracks v3 named (all), B = unnamed strong tracks (OWLv2 >= 0.5, n >= 5).
usage: python review_sheets.py -> clip2_mine/review/{A,B}_NN.jpg"""
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).parent
M = ROOT / "clip2_mine"
names = yaml.safe_load((ROOT / "us_train_v3" / "data.yaml").read_text())["names"]
T = json.loads((M / "tracks.json").read_text())
sets = {"A": [t for t in T if t["v3"] >= 0],
        "B": sorted([t for t in T if t["v3"] < 0 and t["owl"] >= 0.5 and t["n"] >= 5], key=lambda t: t["boxes"][0][0])}
out = M / "review"
out.mkdir(exist_ok=True)
for f in out.glob("*.jpg"):
    f.unlink()
for tag, ts in sets.items():
    tiles = []
    for t in ts:
        f, b = max(t["boxes"], key=lambda fb: (fb[1][2] - fb[1][0]) * (fb[1][3] - fb[1][1]))
        im = cv2.imread(str(M / "frames" / f"f{f:06d}.jpg"))
        x0, y0, x1, y1 = map(int, b[:4])
        m = int(0.8 * max(x1 - x0, y1 - y0)) + 10
        ox, oy = max(0, x0 - m), max(0, y0 - m)
        cr = im[oy:y1 + m, ox:x1 + m].copy()
        cv2.rectangle(cr, (x0 - ox, y0 - oy), (x1 - ox, y1 - oy), (0, 255, 0), 1)
        cr = cv2.resize(cr, (260, 260), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(cr, (0, 0), (260, 34), (0, 0, 0), -1)
        g = f"{t['v3']}:{names[t['v3']][:26]} {t['v3_conf']:.2f}" if t["v3"] >= 0 else f"owl {t['owl']:.2f}"
        cv2.putText(cr, f"#{t['id']} n={t['n']} t={f / 60:.0f}s", (2, 14), 0, 0.45, (0, 255, 255), 1)
        cv2.putText(cr, g, (2, 29), 0, 0.4, (255, 255, 255), 1)
        tiles.append(cr)
    for s in range(0, len(tiles), 40):
        part = tiles[s:s + 40]
        part += [np.zeros_like(tiles[0])] * (-len(part) % 8)
        cv2.imwrite(str(out / f"{tag}_{s // 40:02d}.jpg"),
                    np.vstack([np.hstack(part[i:i + 8]) for i in range(0, len(part), 8)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(tag, len(ts), "tracks", (len(tiles) + 39) // 40, "sheets")
