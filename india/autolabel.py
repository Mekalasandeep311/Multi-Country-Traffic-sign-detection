"""Zero-shot auto-labeling helpers (OWLv2 detector, synthetic sign rendering) for USA traffic signs.

Stage 1  OWLv2 (open-vocabulary detector) finds sign boxes, run on the full frame
         plus overlapping tiles so small, distant signs are not missed.
         Results are cached in <out_dir>/cache so re-runs skip this slow step.
Stage 2  A small classifier on frozen DINOv2 features, trained only on *synthetic*
         crops made from the handbook templates (perspective, scale, blur, night,
         real street backgrounds) plus a "background" class to reject non-signs.
Output   annotated preview images, YOLO .txt labels, and a JSON with top-3 classes
         per box for review.

usage: python autolabel.py [image_dir] [out_dir]   (defaults: this folder, ./output)
"""
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Keep downloaded model weights inside the project folder.
os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "models"))

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, Owlv2ForObjectDetection, Owlv2Processor

ROOT = Path(__file__).parent
OWL_ID = "google/owlv2-base-patch16-ensemble"
DINO_ID = "facebook/dinov2-small"

POS_QUERIES = [
    "a traffic sign", "a road sign", "a circular red traffic sign", "a triangular warning road sign",
    "a blue circular road sign", "a no entry sign", "a speed limit sign", "a yellow chevron sign",
    "a blue rectangular road sign", "a yellow diamond warning sign", "a white rectangular regulatory sign",
    "a green highway guide sign", "a street name sign", "a route marker shield",
]
NEG_QUERIES = [
    "a shop signboard", "a store sign with a brand logo",
    "a traffic light", "a billboard", "a car", "a window", "a street lamp",
]
DET_THRESH = 0.20      # OWLv2 score for a positive query
CLS_THRESH = 0.40      # classifier probability below this -> "sign?" (needs review)
NMS_IOU = 0.4

SYNTH_PER_TEMPLATE = 60
SYNTH_BACKGROUND = 2500
BG_CLASS = 0           # handbook ids are 1..156, so 0 is free for "background"

# Distance: pinhole model, d = f * H_real / h_px. Camera intrinsics are unknown, so
# assume a ~70 deg horizontal FOV and a 0.75 m sign plate. Treat as a rough estimate.
HFOV_DEG = 70.0
SIGN_HEIGHT_M = 0.75

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- stage 1: detection
class SignDetector:
    def __init__(self):
        self.proc = Owlv2Processor.from_pretrained(OWL_ID)
        self.model = Owlv2ForObjectDetection.from_pretrained(OWL_ID).to(DEVICE).eval()
        self.queries = POS_QUERIES + NEG_QUERIES

    @torch.no_grad()
    def _run(self, img: Image.Image):
        inputs = self.proc(text=[self.queries], images=img, return_tensors="pt").to(DEVICE)
        out = self.model(**inputs)
        # OWLv2 pads the image to a square, so boxes are relative to the padded size.
        side = max(img.size)
        res = self.proc.post_process_object_detection(
            out, threshold=0.05, target_sizes=torch.tensor([[side, side]]))[0]
        return res["boxes"].cpu(), res["scores"].cpu(), res["labels"].cpu()

    def __call__(self, img: Image.Image):
        W, H = img.size
        # full frame + 3x2 grid of overlapping tiles
        tiles = [(0, 0, W, H)]
        tw, th = int(W / 2.5), int(H / 1.6)
        for ty in np.linspace(0, H - th, 2).astype(int):
            for tx in np.linspace(0, W - tw, 3).astype(int):
                tiles.append((tx, ty, tx + tw, ty + th))

        boxes, scores, labels = [], [], []
        for x0, y0, x1, y1 in tiles:
            b, s, l = self._run(img.crop((x0, y0, x1, y1)))
            if len(b):
                b = b + torch.tensor([x0, y0, x0, y0])
                boxes.append(b); scores.append(s); labels.append(l)
        if not boxes:
            return []
        boxes, scores, labels = torch.cat(boxes), torch.cat(scores), torch.cat(labels)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H)

        # NMS across *all* queries, so a box that best matches a negative query
        # ("shop signboard", "traffic light") suppresses the positive duplicates.
        keep = torchvision.ops.nms(boxes, scores, NMS_IOU)
        dets = []
        n_pos = len(POS_QUERIES)
        for i in keep.tolist():
            if labels[i] >= n_pos or scores[i] < DET_THRESH:
                continue
            x0, y0, x1, y1 = boxes[i].tolist()
            w, h = x1 - x0, y1 - y0
            if min(w, h) < 10 or max(w, h) > 0.5 * H or not (0.25 < w / h < 4):
                continue
            dets.append({"box": [x0, y0, x1, y1], "det_score": float(scores[i]),
                         "query": self.queries[labels[i]]})
        return dets


# ---------------------------------------------------------------- stage 2: synthetic data
def load_template_rgba(p):
    """Load a handbook graphic and make the white page background around it transparent."""
    rgb = np.array(Image.open(p).convert("RGB"))
    h, w = rgb.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flood = rgb.copy()
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if (rgb[seed[1], seed[0]] > 225).all():
            cv2.floodFill(flood, mask, seed, (0, 0, 0), (12, 12, 12), (12, 12, 12),
                          cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    alpha = 255 - mask[1:-1, 1:-1]
    return np.dstack([rgb, alpha])


def jpeg(im, q):
    buf = io.BytesIO()
    Image.fromarray(im).save(buf, "JPEG", quality=q)
    return np.array(Image.open(buf).convert("RGB"))


def photometric(im, rng):
    im = im.astype(np.float32)
    im = im * rng.uniform(0.25, 1.3) + rng.uniform(-25, 25)              # brightness / night
    mean = im.mean()
    im = (im - mean) * rng.uniform(0.5, 1.2) + mean                       # contrast
    im = im * rng.uniform(0.85, 1.15, size=3)                             # colour cast
    im = im + rng.normal(0, rng.uniform(0, 8), im.shape)                  # sensor noise
    im = np.clip(im, 0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        im = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.3, 1.3))
    return jpeg(im, int(rng.integers(30, 95)))


def warp_sign(tmpl_rgba, target_h, rng):
    """Resize a template to target_h pixels and apply a random perspective/rotation (RGBA out)."""
    h0, w0 = tmpl_rgba.shape[:2]
    s = target_h / h0
    tw, th = max(4, int(w0 * s)), max(4, int(h0 * s))
    sign = cv2.resize(tmpl_rgba, (tw, th), interpolation=cv2.INTER_AREA)

    # perspective + small rotation
    j = 0.12
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    dst = src + rng.uniform(-j, j, (4, 2)).astype(np.float32) * [tw, th]
    ang = math.radians(rng.uniform(-8, 8))
    c, sn = math.cos(ang), math.sin(ang)
    dst = (dst - [tw / 2, th / 2]) @ np.float32([[c, -sn], [sn, c]]) + [tw / 2, th / 2]
    dst -= dst.min(0)
    ow, oh = int(dst[:, 0].max()) + 1, int(dst[:, 1].max()) + 1
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    return cv2.warpPerspective(sign, M, (ow, oh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def synth_sign(tmpl_rgba, backgrounds, rng):
    """One synthetic dashcam-like crop of a sign, as the detector would box it."""
    sign = warp_sign(tmpl_rgba, int(rng.integers(14, 90)), rng)
    oh, ow = sign.shape[:2]

    # paste onto a real street patch, with a jittered box margin like a detector's
    m = rng.uniform(0.02, 0.2)
    cw, ch = int(ow * (1 + 2 * m)) + 2, int(oh * (1 + 2 * m)) + 2
    bg = backgrounds[rng.integers(len(backgrounds))]
    bx, by = rng.integers(0, bg.shape[1] - cw), rng.integers(0, bg.shape[0] - ch)
    canvas = bg[by:by + ch, bx:bx + cw].astype(np.float32).copy()
    ox = int((cw - ow) / 2 + rng.uniform(-0.08, 0.08) * ow)
    oy = int((ch - oh) / 2 + rng.uniform(-0.08, 0.08) * oh)
    ox, oy = np.clip(ox, 0, cw - ow), np.clip(oy, 0, ch - oh)
    a = sign[..., 3:4].astype(np.float32) / 255
    canvas[oy:oy + oh, ox:ox + ow] = canvas[oy:oy + oh, ox:ox + ow] * (1 - a) + sign[..., :3] * a
    return photometric(canvas.astype(np.uint8), rng)


def synth_background(backgrounds, rng):
    bg = backgrounds[rng.integers(len(backgrounds))]
    h = int(rng.integers(14, 110))
    w = int(h * rng.uniform(0.4, 2.5))
    x, y = rng.integers(0, bg.shape[1] - w), rng.integers(0, bg.shape[0] - h)
    return photometric(bg[y:y + h, x:x + w].copy(), rng)


# ---------------------------------------------------------------- stage 2: classifier
class SignClassifier:
    def __init__(self, template_dir: Path, bg_images, cache: Path):
        self.proc = AutoImageProcessor.from_pretrained(DINO_ID)
        self.backbone = AutoModel.from_pretrained(DINO_ID).to(DEVICE).eval()

        templates = sorted(template_dir.glob("*/*.png"))
        key = f"{len(templates)}_{SYNTH_PER_TEMPLATE}_{SYNTH_BACKGROUND}"
        if cache.exists() and torch.load(cache)["key"] == key:
            ck = torch.load(cache)
            print("loaded cached classifier")
        else:
            ck = self._train(templates, bg_images, key)
            torch.save(ck, cache)
        self.class_ids = ck["class_ids"]
        self.head = torch.nn.Linear(ck["W"].shape[1], ck["W"].shape[0])
        self.head.load_state_dict({"weight": ck["W"], "bias": ck["b"]})

    def _train(self, templates, bg_images, key):
        rng = np.random.default_rng(0)
        backgrounds = [np.array(Image.open(p).convert("RGB")) for p in bg_images]
        ims, ys = [], []
        t = time.time()
        for p in templates:
            cid = int(p.parent.name.split("_")[0])
            rgba = load_template_rgba(p)
            for _ in range(SYNTH_PER_TEMPLATE):
                ims.append(synth_sign(rgba, backgrounds, rng)); ys.append(cid)
        for _ in range(SYNTH_BACKGROUND):
            ims.append(synth_background(backgrounds, rng)); ys.append(BG_CLASS)
        print(f"generated {len(ims)} synthetic crops in {time.time() - t:.0f}s")

        sample_dir = ROOT / "output" / "synthetic_samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for i in rng.choice(len(ims), 40, replace=False):
            Image.fromarray(ims[i]).resize((96, 96)).save(sample_dir / f"{ys[i]:03d}_{i}.png")

        t = time.time()
        X = self.embed([Image.fromarray(im) for im in ims])
        print(f"embedded in {time.time() - t:.0f}s")

        class_ids = sorted(set(ys))
        y = torch.tensor([class_ids.index(c) for c in ys])
        X = torch.from_numpy(X)
        head = torch.nn.Linear(X.shape[1], len(class_ids))
        opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-4)
        perm = torch.randperm(len(X))
        n_val = len(X) // 10
        va, tr = perm[:n_val], perm[n_val:]
        for ep in range(300):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(head(X[tr]), y[tr], label_smoothing=0.05)
            loss.backward(); opt.step()
        with torch.no_grad():
            acc = (head(X[va]).argmax(1) == y[va]).float().mean().item()
        print(f"classifier trained: synthetic val acc {acc:.3f}")
        return {"key": key, "class_ids": class_ids,
                "W": head.weight.detach(), "b": head.bias.detach()}

    @torch.no_grad()
    def embed(self, ims, bs=64):
        vecs = []
        for i in range(0, len(ims), bs):
            inp = self.proc(images=ims[i:i + bs], return_tensors="pt").to(DEVICE)
            h = self.backbone(**inp).last_hidden_state
            v = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)   # CLS + mean patch token
            vecs.append(torch.nn.functional.normalize(v, dim=1).cpu())
        return torch.cat(vecs).numpy()

    @torch.no_grad()
    def __call__(self, crops):
        if not crops:
            return []
        probs = self.head(torch.from_numpy(self.embed(crops))).softmax(1)
        top = probs.topk(3, dim=1)
        return [[(self.class_ids[i], float(p)) for p, i in zip(ps.tolist(), ix.tolist())]
                for ps, ix in zip(top.values, top.indices)]


# ---------------------------------------------------------------- drawing
def color_for(cid):
    rng = np.random.default_rng(cid * 7919)
    return tuple(int(c) for c in rng.integers(60, 200, 3))


def draw(img_bgr, dets, header):
    for d in dets:
        x0, y0, x1, y1 = map(int, d["box"])
        col = (90, 90, 90) if d["class_id"] is None else color_for(d["class_id"])
        cv2.rectangle(img_bgr, (x0, y0), (x1, y1), col, 2)
        text = f'{d["class_name"]} {d["prob"]:.2f} {d["dist_m"]:.1f}m'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y0 - 4 if y0 - th - 8 > 0 else y1 + th + 6
        cv2.rectangle(img_bgr, (x0, ty - th - 4), (x0 + tw + 4, ty + 3), col, -1)
        cv2.putText(img_bgr, text, (x0 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img_bgr, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img_bgr, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return img_bgr


# ---------------------------------------------------------------- main
def main(img_dir: Path, out_dir: Path):
    random.seed(0); torch.manual_seed(0)
    classes = {int(k): v for k, v in json.loads((ROOT / "classes.json").read_text()).items()}
    classes.setdefault(85, "Right Turn (for specific vehicles)")
    for sub in ("preview", "labels", "cache"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    # YOLO class index = handbook id - 1
    (out_dir / "classes.txt").write_text("\n".join(classes[i] for i in range(1, max(classes) + 1)))

    images = sorted(img_dir.glob("*.jpg"))
    detector = None  # loaded lazily, only if some image has no cached detections
    classifier = SignClassifier(ROOT / "templates", images, ROOT / "models" / "sign_classifier.pt")

    report = {}
    for p in images:
        img = Image.open(p).convert("RGB")
        W, H = img.size
        cache = out_dir / "cache" / f"{p.stem}.json"
        t0 = time.time()
        if cache.exists():
            dets = json.loads(cache.read_text())
        else:
            detector = detector or SignDetector()
            dets = detector(img)
            cache.write_text(json.dumps(dets))
        det_ms = (time.time() - t0) * 1000

        crops = []
        for d in dets:
            x0, y0, x1, y1 = d["box"]
            m = 0.08 * max(x1 - x0, y1 - y0)
            crops.append(img.crop((max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m))))
        t1 = time.time()
        tops = classifier(crops)
        cls_ms = (time.time() - t1) * 1000

        f_px = (W / 2) / math.tan(math.radians(HFOV_DEG / 2))
        kept, rejected, lines = [], [], []
        for d, top in zip(dets, tops):
            cid, prob = top[0]
            d["top3"] = [(classes.get(c, "background"), round(s, 3)) for c, s in top]
            d["prob"] = prob
            if cid == BG_CLASS:
                rejected.append(d)
                continue
            if prob >= CLS_THRESH:
                d["class_id"], d["class_name"] = int(cid), classes[cid]
            else:
                d["class_id"], d["class_name"] = None, "sign?"
            x0, y0, x1, y1 = d["box"]
            d["dist_m"] = f_px * SIGN_HEIGHT_M / max(1.0, y1 - y0)
            kept.append(d)
            if d["class_id"] is not None:
                cx, cy, bw, bh = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H
                lines.append(f"{d['class_id'] - 1} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (out_dir / "labels" / f"{p.stem}.txt").write_text("\n".join(lines))
        header = f"[owlv2+dinov2 autolabel]  Det: {len(kept)}  Rejected: {len(rejected)}  Classify: {cls_ms:.0f} ms"
        vis = draw(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), kept, header)
        cv2.imwrite(str(out_dir / "preview" / p.name), vis)
        report[p.name] = {"kept": kept, "rejected_as_background": rejected}
        print(f"{p.name}: {len(kept)} signs, {len(rejected)} rejected (detect {det_ms:.0f} ms)", flush=True)
        for d in kept:
            print(f"    {d['class_name']:<40} p={d['prob']:.2f} det={d['det_score']:.2f} top3={d['top3']}", flush=True)

    (out_dir / "detections.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT,
         Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "output")
