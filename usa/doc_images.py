"""4 doc images per country: real labelled sample, sign-swap synthetic, 2 detected video frames."""
import csv, random, collections
from pathlib import Path
import cv2
B = Path("/data/common_datasets/ai24mtech02003")
P = {
 "usa": dict(root=B/"Sandeep_USA", ds="standard_usa", real=["Stop", "Do not enter", "Keep right"], syn=["Merge", "No parking, NY", "Deer crossing"],
             vid="video_test_v3/clip2_detected.mp4", csv="video_test_v3/clip2_detections.csv", off=0.0),
 "japan": dict(root=B/"sandeep_japan", ds="standard_japan", real=["Stop", "Maximum Speed", "No Parking"], syn=["Pedestrian Crossing", "No Entry for Vehicles", "Roundabout"],
             vid="video_test_v2cal/tokya_trimmed_15-30min_detected.mp4", csv="video_test_v2cal/tokya_trimmed_15-30min_detections.csv", off=900.0),
 "singapore": dict(root=B/"sandeep_singapore", ds="sg_signs_balanced", real=["Stop", "No entry", "Keep left"], syn=["Pedestrian crossing ahead", "Roundabout", "Children"],
             vid="video/video1_last30min_detected.mp4", csv="video/video1_last30min_detections.csv", off=0.0),
}
OUT = B/"Sandeep_USA"/"doc_images"
BAD = {"usa": {"No turn on red arrow, Maryland", "Road work ahead", "Road narrows", "Interchange exit direction, Maryland", "Interchange exit direction, NY", "Interchange advance guide, California", "Interchange advance guide, NY", "Reference route marker, New York", "No parking (symbol)", "Give way (Yield)", "Stop here on red, Pennsylvania"},
       "japan": {"Parking Area", "Fire Hydrant"}}
ONLY = {"usa", "japan"}
def names(root):
    return [l.strip() for l in open(root/"classes.txt", encoding="utf-8") if l.strip()]
def boxes(txt, W, H):
    out = []
    for l in open(txt):
        p = l.split()
        if len(p) >= 5:
            c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
            out.append((c, int((cx-w/2)*W), int((cy-h/2)*H), int((cx+w/2)*W), int((cy+h/2)*H)))
    return out
def draw(im, bs, nm):
    t = max(2, im.shape[1] // 600)
    for c, x0, y0, x1, y1 in bs:
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 220, 0), t)
        s = nm[c] if c < len(nm) else str(c)
        fs = 0.5 * t
        (tw, th), _ = cv2.getTextSize(s, 0, fs, t)
        ty = max(th + 6, y0 - 6)
        cv2.rectangle(im, (x0, ty - th - 6), (x0 + tw + 6, ty + 4), (0, 220, 0), -1)
        cv2.putText(im, s, (x0 + 3, ty), 0, fs, (0, 0, 0), t)
    return im
def fit(im):
    s = 1280 / im.shape[1]
    return cv2.resize(im, (1280, int(im.shape[0] * s)))
def pick(ds, prefs, kind, minpx, rnd):
    classes = [d for d in ds.iterdir() if d.is_dir()]
    order = [ds/p for p in prefs if (ds/p).is_dir()] + rnd.sample(classes, len(classes))
    for d in order:
        fs = sorted(d.glob(f"{kind}_*.jpg"))
        rnd.shuffle(fs)
        for f in fs[:60]:
            t = f.with_suffix(".txt")
            if not t.exists():
                continue
            im = cv2.imread(str(f))
            if im is None:
                continue
            bs = boxes(t, im.shape[1], im.shape[0])
            if bs and max(b[3]-b[1] for b in bs) >= minpx:
                return f, im, bs
    raise SystemExit("none " + str(ds))
for k, p in P.items():
    if k not in ONLY:
        continue
    rnd = random.Random(7)
    o = OUT/k; o.mkdir(parents=True, exist_ok=True)
    nm = names(p["root"]); ds = p["root"]/p["ds"]
    f, im, bs = pick(ds, p["real"], "real", 45, rnd)
    cv2.imwrite(str(o/"1_real_sample_labelled.jpg"), fit(draw(im, bs, nm)), [cv2.IMWRITE_JPEG_QUALITY, 88]); print(k, "real", f)
    f, im, bs = pick(ds, p["syn"], "syn", 45, rnd)
    cv2.imwrite(str(o/"2_synthetic_sign_swap.jpg"), fit(draw(im, bs, nm)), [cv2.IMWRITE_JPEG_QUALITY, 88]); print(k, "syn", f)
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(p["root"]/p["csv"])):
        by[float(r["time_s"])].append(r)
    def score(rs):
        big = [r for r in rs if float(r["y1"]) - float(r["y0"]) >= 28]
        return (len({r["class_name"] for r in big}), sum(float(r["conf"]) for r in big))
    bad = BAD.get(k, set())
    ts = sorted((t for t in by if not any(r["class_name"] in bad for r in by[t])), key=lambda t: score(by[t]), reverse=True)
    chosen = []
    for t in ts:
        if all(abs(t - u) > 60 for u in chosen):
            chosen.append(t)
        if len(chosen) == 2:
            break
    cap = cv2.VideoCapture(str(p["root"]/p["vid"])); fps = cap.get(cv2.CAP_PROP_FPS)
    for i, t in enumerate(chosen):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round((t - p["off"]) * fps)))
        ok, im = cap.read()
        cv2.imwrite(str(o/f"{3+i}_detected_frame_{'ab'[i]}.jpg"), fit(im), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(k, "det", t, [r["class_name"] for r in by[t]])
