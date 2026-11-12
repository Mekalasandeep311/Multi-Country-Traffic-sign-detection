"""Audit every labelled box against the 98 rule-book pictures with DINOv2 embeddings -> audit.csv.
Per box: crop similarity to every template (rank of its own class, best class), similarity to its own template
mirrored (mirrored copies), NCC edge check vs mirrored template, box area share and white border share
(clip-art / stock pictures). Suggested class for P-sign and informatory-board boxes among their candidates."""
import csv, json, os
from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
os.environ.setdefault("HF_HOME", "/data/common_datasets/ai24mtech02003/Sandeep_USA/models")
from transformers import AutoModel
R = Path(__file__).parent
CM = json.load(open(R / "class_map.json"))
C = CM["classes"]
cand = {"P": CM["P_classes"], "INFO": CM["INFO_classes"]}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
model = AutoModel.from_pretrained("facebook/dinov2-base", local_files_only=True).cuda().eval().half()

def emb(rgbs):
    x = torch.from_numpy(np.stack([cv2.resize(np.ascontiguousarray(r), (224, 224)) for r in rgbs])).cuda().permute(0, 3, 1, 2).float() / 255
    x = ((x - MEAN) / STD).half()
    with torch.no_grad():
        f = model(pixel_values=x).pooler_output.float()
    return torch.nn.functional.normalize(f, dim=1)

def load_rgb(p):
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im.ndim == 3 and im.shape[2] == 4:
        a = im[:, :, 3:4] / 255.0
        im = (im[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

tdirs = sorted((R / "templates").iterdir())
T = [load_rgb(next(d.glob("handbook.*"))) for d in tdirs]
TE = emb(T); TM = emb([t[:, ::-1].copy() for t in T])
tsym = (TE * TM).sum(1).cpu().numpy()            # how mirror-symmetric each template is
TG = [cv2.Canny(cv2.resize(cv2.cvtColor(t, cv2.COLOR_RGB2GRAY), (64, 64)), 60, 160).astype(np.float32) for t in T]

items = []
for l in open(R / "inventory.jsonl"):
    r = json.loads(l)
    for k, b in enumerate(r.get("boxes", [])):
        if b[1] is not None:
            items.append((r["path"], r["stem"], r["kind"], k, b))

class DS(Dataset):
    def __len__(self): return len(items)
    def __getitem__(self, i):
        path, stem, kind, k, (o, n, cx, cy, w, h) = items[i]
        im = load_rgb(path); H, W = im.shape[:2]
        x0, y0, x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H), int((cx + w / 2) * W), int((cy + h / 2) * H)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, max(x1, x0 + 2)), min(H, max(y1, y0 + 2))
        cr = im[y0:y1, x0:x1]
        ring = np.concatenate([im[max(0, y0 - 8):y0, x0:x1].reshape(-1, 3), im[y1:y1 + 8, x0:x1].reshape(-1, 3),
                               im[y0:y1, max(0, x0 - 8):x0].reshape(-1, 3), im[y0:y1, x1:x1 + 8].reshape(-1, 3)])
        white = float((ring.min(1) > 230).mean()) if len(ring) else 1.0
        g = cv2.Canny(cv2.resize(cv2.cvtColor(cr, cv2.COLOR_RGB2GRAY), (64, 64)), 60, 160).astype(np.float32)
        return cv2.resize(cr, (224, 224)), g, i, w * h, white, x1 - x0, y1 - y0

def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))

out = open(R / "audit.csv", "w", newline=""); wr = csv.writer(out)
wr.writerow(["stem", "path", "kind", "box", "old", "new", "own_rank", "own_sim", "own_mirror_sim", "tmpl_sym",
             "ncc", "ncc_mirror", "best", "best_sim", "suggest", "area", "white", "wpx", "hpx"])
dl = DataLoader(DS(), batch_size=128, num_workers=12, collate_fn=lambda b: list(zip(*b)))
for bi, (crops, gs, idx, area, white, wpx, hpx) in enumerate(dl):
    E = emb(list(crops)); S = (E @ TE.T).cpu().numpy(); SM = (E @ TM.T).cpu().numpy()
    for j, i in enumerate(idx):
        path, stem, kind, k, (o, n, *_rest) = items[i]
        best = int(S[j].argmax())
        if isinstance(n, int) and n < 98:
            own_rank = int((S[j] > S[j, n]).sum()) + 1
            row = [own_rank, f"{S[j, n]:.3f}", f"{SM[j, n]:.3f}", f"{tsym[n]:.3f}",
                   f"{ncc(gs[j], TG[n]):.3f}", f"{ncc(gs[j], TG[n][:, ::-1].copy()):.3f}", best, f"{S[j, best]:.3f}", ""]
        else:
            cs = cand.get(n, [])
            sug = max(cs, key=lambda c: S[j, c]) if cs else ""
            row = ["", "", "", "", "", "", best, f"{S[j, best]:.3f}", sug]
        wr.writerow([stem, path, kind, k, o, n] + row + [f"{area[j]:.4f}", f"{white[j]:.2f}", wpx[j], hpx[j]])
    if bi % 20 == 0:
        print(bi * 128, "/", len(items), flush=True)
out.close(); print("done", len(items))
