"""Remove track drift: every box of a labelled track must look like the reviewed crop (the track's largest boxes), by DINOv2.
Boxes below the threshold become grey in mine_india.py write. -> india_mine/<vid>/keep.json {track: [frames kept]}."""
import json, os, sys
from pathlib import Path
import cv2, numpy as np, torch
os.environ.setdefault("HF_HOME", "/data/common_datasets/ai24mtech02003/Sandeep_USA/models")
from transformers import AutoModel
m = AutoModel.from_pretrained("facebook/dinov2-base", local_files_only=True).cuda().eval().half()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
def emb(cs):
    x = torch.from_numpy(np.stack(cs)).cuda().permute(0, 3, 1, 2).float() / 255
    with torch.no_grad():
        f = m(pixel_values=((x - MEAN) / STD).half()).pooler_output.float()
    return torch.nn.functional.normalize(f, dim=1)
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.55
for v in ("cam_20261106_100001", "cam_20261106_100849"):
    O = Path("india_mine") / v
    dec = dict(l.split() for l in open(O / "decisions.txt"))
    keep = {}; tot = kept = 0; hist = []
    for t in json.load(open(O / "tracks.json")):
        if dec[str(t["id"])] in ("n", "g"): continue
        crops = []
        for f, b in t["boxes"]:
            im = cv2.imread(str(O / "frames" / f"f{f:06d}.jpg")); x0, y0, x1, y1 = (max(0, int(a)) for a in b[:4])
            crops.append(cv2.cvtColor(cv2.resize(im[y0:max(y1, y0 + 2), x0:max(x1, x0 + 2)], (224, 224)), cv2.COLOR_BGR2RGB))
        E = torch.cat([emb(crops[i:i + 64]) for i in range(0, len(crops), 64)])
        area = [(b[2] - b[0]) * (b[3] - b[1]) for _, b in t["boxes"]]
        top = np.argsort(area)[-3:]; ref = torch.nn.functional.normalize(E[top].mean(0), dim=0)
        s = (E @ ref).cpu().numpy(); hist += s.tolist()
        keep[str(t["id"])] = [t["boxes"][i][0] for i in range(len(s)) if s[i] >= THR]
        tot += len(s); kept += len(keep[str(t["id"])])
    json.dump(keep, open(O / "keep.json", "w"))
    print(v, "boxes", tot, "kept", kept, "sim pct10/50/90", np.percentile(hist, [10, 50, 90]).round(2), flush=True)
