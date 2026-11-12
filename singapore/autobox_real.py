"""Auto-box the folder's sign in every real image (YOLO .txt next to each image).

Each class folder holds frames that all contain sign C. Per folder:
  A. candidates  OWLv2 on full frame + 6 tiles (batched, GPU preprocessing). For every box keep
                 its sign-ness (max score over generic sign queries) and its OWLv2 class embedding.
  B. class match cosine between each box embedding and
                 - image queries: OWLv2 embeddings of the handbook template(s) of C (one-shot,
                   like OWLv2 image-guided detection), and
                 - a text query: "a <class name> road sign".
                 Only boxes that look like a sign (sign-ness >= SIGN_GATE) can be picked, so
                 windows / walls / trees cannot win.
  C. refine      the confident picks of the folder (clear margin over the runner-up) give a
                 real-appearance prototype of C; all images are re-scored with it.
  D. accept      score >= max(ABS_MIN, median - K_MAD * MAD); further instances of C in the same
                 frame are added when they score close to the best one.
Outputs
  Traffic Signs Singapore/<class>/<img>.txt          YOLO label(s) of the folder's sign (accepted only)
  autobox/<class>.json                                all picks + scores + status
  autobox/review/<class>_accepted.jpg|_rejected.jpg   contact sheets (lowest scores first)
  autobox/report.csv

usage: python autobox_real.py [phase]   phase: all (default) | select  (select reuses cached A)
       ONLY="Slow,No U-Turn" python autobox_real.py      # subset of folders
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "models"))

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn
import torchvision
import yaml
from PIL import Image

from autolabel import OWL_ID, Owlv2ForObjectDetection, Owlv2Processor, load_template_rgba

ROOT = Path(__file__).parent
DATA = ROOT / "Traffic Signs Singapore"
OUT = ROOT / "autobox"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIGN_QUERIES = [
    "a traffic sign", "a road sign", "a circular red traffic sign", "a triangular warning road sign",
    "a blue circular road sign", "a no entry sign", "a speed limit sign", "a yellow chevron sign",
    "a blue rectangular road sign", "a green direction sign board", "an orange road works sign",
    "an electronic traffic display board", "a white rectangular sign with text", "a red rectangular sign",
]
CAND_THRESH = 0.05      # keep boxes this sign-like as candidates
SIGN_GATE = 0.12        # a pick must be at least this sign-like
NMS_IOU = 0.5
ABS_MIN = 0.05        # floor on P(class | box)
K_MAD = 3.0
CONS_ACCEPT_MAD = 1.5   # consistency method: accept only picks close to the folder's typical score
ITERS = 3
MULTI_GAP = 0.03
OWL_SIZE = 960
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class Owl:
    def __init__(self):
        self.proc = Owlv2Processor.from_pretrained(OWL_ID)
        self.model = Owlv2ForObjectDetection.from_pretrained(OWL_ID).to(DEVICE).eval().half()
        self.mean, self.std = CLIP_MEAN.to(DEVICE).half(), CLIP_STD.to(DEVICE).half()
        self.sign_text = self.text(SIGN_QUERIES)

    def text(self, queries):
        t = self.proc(text=[queries], return_tensors="pt", padding="max_length", truncation=True, max_length=16)
        return {k: v.to(DEVICE) for k, v in t.items()}

    def prep(self, crops):
        """list of HxWx3 uint8 RGB -> (B,3,960,960) like Owlv2Processor (pad bottom/right with gray)."""
        out = []
        for c in crops:
            x = torch.from_numpy(c).to(DEVICE).permute(2, 0, 1).half() / 255
            h, w = x.shape[1:]
            s = max(h, w)
            canvas = torch.full((3, s, s), 0.5, device=DEVICE, dtype=x.dtype)
            canvas[:, :h, :w] = x
            out.append(Fn.interpolate(canvas[None], size=(OWL_SIZE, OWL_SIZE), mode="bilinear",
                                      align_corners=False, antialias=True)[0])
        return (torch.stack(out) - self.mean) / self.std

    @torch.no_grad()
    def forward(self, crops, text):
        px = self.prep(crops)
        n = len(crops)
        ids = text["input_ids"].repeat(n, 1)
        am = text["attention_mask"].repeat(n, 1)
        o = self.model(input_ids=ids, attention_mask=am, pixel_values=px)
        # boxes are cx,cy,w,h relative to the padded square
        return o.logits.float().sigmoid(), o.pred_boxes.float(), o.class_embeds.float(), o.text_embeds.float()

    def candidates(self, rgb):
        H, W = rgb.shape[:2]
        tiles = [(0, 0, W, H)]
        tw, th = int(W / 2.5), int(H / 1.6)
        for ty in np.linspace(0, H - th, 2).astype(int):
            for tx in np.linspace(0, W - tw, 3).astype(int):
                tiles.append((tx, ty, tx + tw, ty + th))
        probs, pboxes, emb, _ = self.forward([rgb[y0:y1, x0:x1] for x0, y0, x1, y1 in tiles], self.sign_text)
        B, S, E = [], [], []
        for t, (x0, y0, x1, y1) in enumerate(tiles):
            s = probs[t].max(1).values
            keep = s >= CAND_THRESH
            if not keep.any():
                continue
            side = max(x1 - x0, y1 - y0)
            cx, cy, w, h = (pboxes[t][keep] * side).unbind(1)
            B.append(torch.stack([cx - w / 2 + x0, cy - h / 2 + y0, cx + w / 2 + x0, cy + h / 2 + y0], 1))
            S.append(s[keep]); E.append(emb[t][keep])
        if not B:
            return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 512), np.float32)
        B, S, E = torch.cat(B), torch.cat(S), torch.cat(E)
        B[:, 0::2] = B[:, 0::2].clamp(0, W); B[:, 1::2] = B[:, 1::2].clamp(0, H)
        k = torchvision.ops.nms(B, S, NMS_IOU)
        B, S, E = B[k], S[k], E[k]
        w, h = B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]
        ok = (torch.minimum(w, h) >= 8) & (h < 0.8 * H) & (w < 0.8 * W) & (w / h > 0.15) & (w / h < 7)
        return B[ok].cpu().numpy(), S[ok].cpu().numpy(), Fn.normalize(E[ok], dim=1).cpu().numpy()

    def template_queries(self, pngs, rng):
        """OWLv2 box embeddings of the template pasted at several sizes on gray / noisy backgrounds."""
        qs = []
        for p in pngs:
            rgba = load_template_rgba(p)
            for size, bg in [(320, 128), (160, 90), (80, 160), (48, 110)]:
                canvas = np.full((480, 480, 3), bg, np.uint8)
                canvas = np.clip(canvas + rng.normal(0, 12, canvas.shape), 0, 255).astype(np.uint8)
                s = size / max(rgba.shape[:2])
                t = cv2.resize(rgba, (max(4, int(rgba.shape[1] * s)), max(4, int(rgba.shape[0] * s))),
                               interpolation=cv2.INTER_AREA)
                th, tw = t.shape[:2]
                y, x = (480 - th) // 2, (480 - tw) // 2
                a = t[..., 3:4] / 255
                canvas[y:y + th, x:x + tw] = (canvas[y:y + th, x:x + tw] * (1 - a) + t[..., :3] * a).astype(np.uint8)
                _, pb, emb, _ = self.forward([canvas], self.sign_text)
                gt = torch.tensor([[x, y, x + tw, y + th]], dtype=torch.float32, device=DEVICE) / 480
                cx, cy, w, h = pb[0].unbind(1)
                boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
                qs.append(Fn.normalize(emb[0][torchvision.ops.box_iou(boxes, gt)[:, 0].argmax()], dim=0))
        return torch.stack(qs).cpu().numpy() if qs else None

    def text_query_raw(self, text):
        _, _, _, te = self.forward([np.zeros((32, 32, 3), np.uint8)], self.text([text]))
        return Fn.normalize(te[0, 0], dim=0).cpu().numpy()


def extract_folder(folder, owl):
    files = sorted(f for f in os.listdir(DATA / folder) if f.lower().endswith(".jpg"))
    idx, boxes, sign, emb = [], [], [], []
    for i, f in enumerate(files):
        rgb = np.array(Image.open(DATA / folder / f).convert("RGB"))
        b, s, e = owl.candidates(rgb)
        idx += [i] * len(b); boxes.append(b); sign.append(s); emb.append(e)
    np.savez_compressed(OUT / "cache" / f"{folder}.npz", files=np.array(files), img_idx=np.array(idx, int),
                        boxes=np.concatenate(boxes), sign=np.concatenate(sign), emb=np.concatenate(emb))


BACKGROUND_TEXT = ["a traffic light", "a street name sign", "a shop signboard", "a billboard",
                   "a building window", "a car", "a bus", "a lamp post", "a tree", "a road marking"]
TEMPERATURE = 0.02     # softmax temperature over class cosines
TOP_K = 3              # class C must be among the top-K classes of the chosen box


class ClassBank:
    """Image (template) and text query embeddings for every class + background texts."""

    def __init__(self, owl, names):
        cache = OUT / "queries.npz"
        if cache.exists():
            z = np.load(cache)
            self.img, self.img_cls, self.txt = z["img"], z["img_cls"], z["txt"]
        else:
            img, img_cls = [], []
            for cid in range(156):
                pngs = [p for d in (ROOT / "templates").glob(f"{cid + 1:03d}_*") for p in d.glob("*.png")]
                q = owl.template_queries(pngs, np.random.default_rng(cid))
                if q is not None:
                    img.append(q); img_cls += [cid] * len(q)
            self.img, self.img_cls = np.concatenate(img), np.array(img_cls)
            texts = [f"a {names[k]} road sign" for k in range(len(names))] + BACKGROUND_TEXT
            self.txt = np.stack([owl.text_query_raw(t) for t in texts])
            np.savez(cache, img=self.img, img_cls=self.img_cls, txt=self.txt)
        self.n_cls = len(names)

    def probs(self, E, extra=None):
        """P(class | box) from template cosines and from text cosines; columns = classes (+ background
        for text). `extra` = {cid: matrix of additional image queries (e.g. real prototypes)}."""
        S = E @ self.img.T
        img = np.full((len(E), self.n_cls), -1.0)
        for c in np.unique(self.img_cls):
            img[:, c] = S[:, self.img_cls == c].max(1)
        for c, q in (extra or {}).items():
            img[:, c] = np.maximum(img[:, c], (E @ q.T).max(1))
        has = img.max(0) > -1
        p_img = np.zeros_like(img)
        x = img[:, has] / TEMPERATURE
        x = np.exp(x - x.max(1, keepdims=True))
        p_img[:, has] = x / x.sum(1, keepdims=True)
        t = (E @ self.txt.T) / TEMPERATURE
        t = np.exp(t - t.max(1, keepdims=True))
        p_txt = t / t.sum(1, keepdims=True)
        return p_img, p_txt, has


YOLO_WEIGHTS = ROOT / "runs" / "sg_signs" / "weights" / "best.pt"
YOLO_ACCEPT = 0.30     # confident: label written now
YOLO_UNCERTAIN = 0.10  # 0.10-0.30: kept as "uncertain" (no .txt) for round 2 with the retrained model
FALLBACK_BELOW = 0.30  # if YOLO accepts < 30% of a folder, use the folder-consistency method instead
YOLO_TOPK = 300        # anchors kept per image for class C


class YoloClassScores:
    """Raw per-anchor score of ONE class from the synthetic-trained YOLO (no NMS, no argmax)."""

    def __init__(self):
        from ultralytics import YOLO
        self.net = YOLO(str(YOLO_WEIGHTS)).model.to(DEVICE).eval().half()
        self.nc = len(self.net.names)

    @torch.no_grad()
    def __call__(self, rgb, cid):
        H, W = rgb.shape[:2]
        Hp, Wp = -(-H // 32) * 32, -(-W // 32) * 32
        x = torch.zeros((1, 3, Hp, Wp), device=DEVICE, dtype=torch.half)
        x[0, :, :H, :W] = torch.from_numpy(rgb).to(DEVICE).permute(2, 0, 1).half() / 255
        out = self.net(x)
        pred = (out[0] if isinstance(out, (list, tuple)) else out)[0].float()   # (4+nc, N)
        xywh, cls = pred[:4].T, pred[4:].T
        sc = cls[:, cid]
        k = sc.topk(min(YOLO_TOPK, len(sc))).indices
        b = xywh[k]
        boxes = torch.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        return boxes.cpu().numpy(), sc[k].cpu().numpy()


def extract_yolo(folder, cid, yolo):
    files = sorted(f for f in os.listdir(DATA / folder) if f.lower().endswith(".jpg"))
    B, S = [], []
    for f in files:
        b, s = yolo(np.array(Image.open(DATA / folder / f).convert("RGB")), cid)
        B.append(b); S.append(s)
    np.savez_compressed(OUT / "cache" / f"{folder}_yolo.npz", boxes=np.stack(B), scores=np.stack(S))


def iou_matrix(a, b):
    return torchvision.ops.box_iou(torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)).numpy()


def select(folder, cid, name, owl, bank):
    z = np.load(OUT / "cache" / f"{folder}.npz")
    files, idx, boxes, sign, E = z["files"], z["img_idx"], z["boxes"], z["sign"], z["emb"]
    n = len(files)
    yolo_known = cid < 156 and bool((bank.img_cls == cid).any())
    if not yolo_known:
        return select_consistency(folder, cid, name, owl, files, idx, boxes, sign, E)

    y = np.load(OUT / "cache" / f"{folder}_yolo.npz")
    yb, ys = y["boxes"], y["scores"]
    tsim = (E @ bank.img[bank.img_cls == cid].T).max(1)          # OWL one-shot similarity (tie-break)

    result, best_scores = {}, []
    for i in range(n):
        f = str(files[i])
        # candidates: sign-like OWL boxes + YOLO's own top boxes for class C (after NMS)
        o = np.nonzero((idx == i) & (sign >= SIGN_GATE))[0]
        cand = [boxes[j] for j in o]
        owl_sim = [tsim[j] for j in o]
        keep = torchvision.ops.nms(torch.tensor(yb[i]), torch.tensor(ys[i]), 0.5).numpy()
        keep = [k for k in keep if ys[i][k] >= YOLO_UNCERTAIN / 2][:10]
        cand += [yb[i][k] for k in keep]
        owl_sim += [0.0] * len(keep)
        if not cand:
            result[f] = {"status": "rejected", "reason": "no candidates", "score": None, "boxes": []}
            continue
        cand = np.array(cand)
        ov = iou_matrix(cand, yb[i])
        yc = np.where(ov >= 0.3, ys[i][None], 0).max(1)              # YOLO class-C score at each candidate
        fused = yc + 0.05 * np.array(owl_sim)
        order = np.argsort(-fused)
        top = order[0]
        s = float(yc[top])
        best_scores.append(s)
        if s < YOLO_ACCEPT:
            result[f] = {"status": "uncertain" if s >= YOLO_UNCERTAIN else "rejected",
                         "reason": "low class score", "score": s, "boxes": [cand[top].round(1).tolist()]}
            continue
        chosen = [top]
        for j in order[1:]:
            if yc[j] >= max(YOLO_ACCEPT, 0.6 * s) and len(chosen) < 4 and all(
                    iou_matrix(cand[[j]], cand[[k]])[0, 0] < 0.3 for k in chosen):
                chosen.append(j)
        result[f] = {"status": "accepted", "score": s, "boxes": [cand[j].round(1).tolist() for j in chosen]}
    med = float(np.median(best_scores)) if best_scores else 0.0
    if sum(r["status"] == "accepted" for r in result.values()) < FALLBACK_BELOW * n:
        alt, info = select_consistency(folder, cid, name, owl, files, idx, boxes, sign, E)
        if sum(r["status"] == "accepted" for r in alt.values()) > sum(r["status"] == "accepted" for r in result.values()):
            info["method"] = "consistency (fallback)"
            return alt, info
    return result, {"threshold": YOLO_ACCEPT, "median": med, "has_template": True, "seeds": 0, "method": "yolo"}


def select_consistency(folder, cid, name, owl, files, idx, boxes, sign, E):
    """Classes the YOLO does not know (no handbook template): the sign that recurs across the
    folder + text query "a <name> road sign" decides."""
    n = len(files)
    per = [np.nonzero((idx == i) & (sign >= SIGN_GATE))[0] for i in range(n)]
    txt = owl.text_query_raw(f"a {name} road sign")
    zf = lambda v: (v - v.mean()) / (v.std() + 1e-6)
    rng = np.random.default_rng(0)
    refs = [p for p in per if len(p)]
    if not refs:
        return ({str(f): {"status": "rejected", "reason": "no sign-like box", "score": None, "boxes": []} for f in files},
                {"threshold": 0, "median": 0, "has_template": False, "seeds": 0})
    refs = [refs[k] for k in rng.choice(len(refs), min(120, len(refs)), replace=False)]
    consist = np.mean([(E @ E[r].T).max(1) for r in refs], axis=0)
    zt = zf(E @ txt)
    score = zf(consist) + zt
    pick = {i: p[np.argmax(score[p])] for i, p in enumerate(per) if len(p)}
    for _ in range(ITERS):
        sel = np.array(list(pick.values()))
        proto = E[sel].mean(0); proto /= np.linalg.norm(proto)
        good = sel[E[sel] @ proto >= np.quantile(E[sel] @ proto, 0.3)]
        proto = E[good].mean(0); proto /= np.linalg.norm(proto)
        sims = E @ proto
        pick = {i: p[np.argmax(sims[p] + 0.03 * zt[p])] for i, p in enumerate(per) if len(p)}
    best = np.array([sims[pick[i]] if i in pick else -1.0 for i in range(n)])
    valid = best[best > -1]
    med = float(np.median(valid)); mad = float(np.median(np.abs(valid - med)))
    thr = max(0.5, med - CONS_ACCEPT_MAD * 1.4826 * mad)          # confident
    thr_unc = max(0.4, med - K_MAD * 1.4826 * mad)                 # below this: rejected
    result = {}
    for i in range(n):
        f = str(files[i])
        if i not in pick:
            result[f] = {"status": "rejected", "reason": "no sign-like box", "score": None, "boxes": []}
        elif best[i] < thr:
            result[f] = {"status": "uncertain" if best[i] >= thr_unc else "rejected", "reason": "low score",
                         "score": float(best[i]), "boxes": [boxes[pick[i]].round(1).tolist()]}
        else:
            result[f] = {"status": "accepted", "score": float(best[i]), "boxes": [boxes[pick[i]].round(1).tolist()]}
    return result, {"threshold": thr, "median": med, "has_template": False, "seeds": 0, "method": "consistency"}


def crop_of(img, box, pad=0.1):
    x0, y0, x1, y1 = box
    m = pad * max(x1 - x0, y1 - y0)
    H, W = img.shape[:2]
    return img[int(max(0, y0 - m)):int(min(H, y1 + m)), int(max(0, x0 - m)):int(min(W, x1 + m))]


def write_outputs(folder, cid, result):
    for f, r in result.items():
        txt = DATA / folder / (Path(f).stem + ".txt")
        if r["status"] != "accepted":
            if txt.exists():
                txt.unlink()          # the dataset had no .txt files; this only removes our own earlier output
            continue
        with Image.open(DATA / folder / f) as im:
            W, H = im.size
        txt.write_text("".join(f"{cid} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                               f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n" for x0, y0, x1, y1 in r["boxes"]))
    (OUT / f"{folder}.json").write_text(json.dumps(result, indent=1))

    for status in ("accepted", "uncertain", "rejected"):
        items = [(f, r) for f, r in result.items() if r["status"] == status and r["boxes"]]
        items = sorted(items, key=lambda fr: fr[1]["score"] or 0)
        if len(items) > 80:                      # lowest 40 + evenly spread rest
            items = items[:40] + [items[k] for k in np.linspace(40, len(items) - 1, 40).astype(int)]
        tiles = []
        for f, r in items:
            im = cv2.imread(str(DATA / folder / f))
            c = cv2.resize(crop_of(im, r["boxes"][0], pad=0.6), (128, 128))
            cv2.rectangle(c, (0, 0), (127, 127), {"accepted": (0, 200, 0), "uncertain": (0, 200, 255)}.get(status, (0, 0, 220)), 1)
            cv2.putText(c, f"{r['score']:.2f}", (3, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            tiles.append(c)
        if tiles:
            tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 10)
            cv2.imwrite(str(OUT / "review" / f"{folder}_{status}.jpg"),
                        np.vstack([np.hstack(tiles[k:k + 10]) for k in range(0, len(tiles), 10)]))


def main(phase="all"):
    cmap = yaml.safe_load((ROOT / "class_map.yaml").read_text())
    folders, names = cmap["folders"], cmap["names"]
    if os.environ.get("ONLY"):
        folders = {f: folders[f] for f in os.environ["ONLY"].split(",")}
    for sub in ("cache", "review"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    owl = Owl()
    if phase == "all":
        for k, folder in enumerate(sorted(folders, key=lambda f: len(os.listdir(DATA / f)))):
            if (OUT / "cache" / f"{folder}.npz").exists():
                continue
            t = time.time()
            extract_folder(folder, owl)
            print(f"[{k + 1}/{len(folders)}] candidates {folder}: {time.time() - t:.0f}s", flush=True)

    bank = ClassBank(owl, names)
    yolo = None
    for folder, cid in folders.items():
        if cid < 156 and bool((bank.img_cls == cid).any()) and not (OUT / "cache" / f"{folder}_yolo.npz").exists():
            yolo = yolo or YoloClassScores()
            extract_yolo(folder, cid, yolo)
    rows = []
    for folder, cid in folders.items():
        result, info = select(folder, cid, names[cid], owl, bank)
        write_outputs(folder, cid, result)
        acc = sum(r["status"] == "accepted" for r in result.values())
        unc = sum(r["status"] == "uncertain" for r in result.values())
        rows.append([folder, cid, len(result), acc, unc, len(result) - acc - unc, round(info["threshold"], 3),
                     round(info["median"], 3), info["has_template"], info.get("method", "")])
        print(f"{folder}: {acc}/{len(result)} accepted, {unc} uncertain ({info.get('method')}, "
              f"median {info['median']:.2f})", flush=True)
    report = OUT / "report.csv"
    old = {}
    if report.exists():
        with open(report) as fh:
            old = {r[0]: r for r in csv.reader(fh)}
    for r in rows:
        old[r[0]] = r
    old.pop("class", None)
    with open(report, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "class_id", "images", "accepted", "uncertain", "rejected", "threshold", "median_score",
                    "template", "method"])
        w.writerows(sorted(old.values()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
