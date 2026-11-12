"""Data for india_v2 (fine-tune from v1):
- new classes 100 Charging station, 101 Width limit: 400 sign-swap images each (video-cut templates; Width limit also
  drawn with other distances so the model does not learn only "3.5 m"); 200 extra Tunnel images with the red-triangle template.
- india_train_v2/: train = v1 train + new images + test-bed frames; val = real val + test-bed val blocks;
  test = real test photos; test_bed = held-out test-bed time blocks (never trained on). Hold-out images are copies,
  never links to the read-only source.
usage: python build_v2_data.py <workers>"""
import json, os, random, shutil, sys
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np
import build_standard as B

R = Path(__file__).parent
C = json.load(open(R / "class_map.json"))["classes"]
D = R / "india_train_v2"
BLOCK = 600        # 30 s blocks at 20 fps
GAP = 40           # train frames within 2 s of a held-out block are dropped


def width_variants():
    out = [B.cutout(next((R / "templates" / "101_Width limit").glob("handbook.*")))]
    for txt in ("2.5m", "3m", "4m", "4.5m", "5.5m"):
        S = 400; im = np.full((S, S, 3), 255, np.uint8); a = np.zeros((S, S), np.uint8)
        cv2.circle(a, (S // 2, S // 2), S // 2 - 2, 255, -1, cv2.LINE_AA)
        cv2.circle(im, (S // 2, S // 2), S // 2 - 2, (40, 30, 200), -1, cv2.LINE_AA)
        cv2.circle(im, (S // 2, S // 2), int(S * 0.39), (235, 235, 235), -1, cv2.LINE_AA)
        fs = 3.0 if len(txt) <= 2 else 2.3
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 9)
        cv2.putText(im, txt, ((S - tw) // 2, (S + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), 9, cv2.LINE_AA)
        for sx in (1, -1):               # arrows pointing inwards from left and right
            x = S // 2 - sx * int(S * 0.39)
            cv2.fillPoly(im, [np.array([[x, S // 2 - 26], [x, S // 2 + 26], [x + sx * 34, S // 2]])], (20, 20, 20), cv2.LINE_AA)
        out.append((im, cv2.GaussianBlur(a, (3, 3), 0)))
    return out


def gen(job):
    c, tmpl, n, prefix, folder = job
    rnd = random.Random(5000 + c + len(prefix)); d = B.OUT / folder; d.mkdir(parents=True, exist_ok=True); k = 0
    while k < n:
        B.T[c] = rnd.choice(tmpl)
        img, lab = B.synth(c, rnd)
        if img is None:
            continue
        cv2.imwrite(str(d / f"{prefix}{k:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (d / f"{prefix}{k:05d}.txt").write_text("".join(lab)); k += 1
    return folder, k


def copy_small(src, dst):
    im = cv2.imread(str(src)); s = 1920 / max(im.shape[:2])
    if s < 1:
        im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main():
    for i in (100, 101):
        B.T[i] = B.cutout(next((R / "templates").glob(f"{i:03d}_*/handbook.*")))
    tun = B.cutout(R / "templates_extra" / "075_Tunnel triangle" / "handbook.png")
    jobs = [(100, [B.T[100]], 400, "syn_", "100_Charging station"), (101, width_variants(), 400, "syn_", "101_Width limit"),
            (75, [tun], 200, "syn_tb_", next(p.name for p in B.OUT.glob("075_*")))]
    with Pool(int(sys.argv[1])) as p:
        print(p.map(gen, jobs), flush=True)

    for s in ("train", "val", "test", "test_bed", "test_v1", "test_bed_v1"):
        for k in ("images", "labels"):
            (D / k / s).mkdir(parents=True, exist_ok=True)
    src = R / "india_train"
    n = 0
    for f in os.scandir(src / "images" / "train"):               # v1 train (links to the v1 dataset)
        os.symlink(os.path.realpath(f.path), D / "images" / "train" / f.name)
        shutil.copy(src / "labels" / "train" / (Path(f.name).stem + ".txt"), D / "labels" / "train"); n += 1
    for name, (c, *_ ) in (("100_Charging station", (100,)), ("101_Width limit", (101,))):
        for im in (B.OUT / name).glob("*.jpg"):
            os.symlink(im, D / "images" / "train" / f"{c}_{im.name}"); shutil.copy(im.with_suffix(".txt"), D / "labels" / "train" / f"{c}_{im.stem}.txt"); n += 1
    for im in next(B.OUT.glob("075_*")).glob("syn_tb_*.jpg"):
        os.symlink(im, D / "images" / "train" / f"075_{im.name}"); shutil.copy(im.with_suffix(".txt"), D / "labels" / "train" / f"075_{im.stem}.txt"); n += 1
    print("train from dataset", n, flush=True)

    for s, dest in (("val", "val"), ("test", "test")):            # real hold-out: copies
        for f in os.scandir(src / "images" / s):
            copy_small(f.path, D / "images" / dest / (Path(f.name).stem + ".jpg"))
            shutil.copy(src / "labels" / s / (Path(f.name).stem + ".txt"), D / "labels" / dest)
    for f in os.scandir(D / "images" / "test"):                  # v1 has no class 100 / 101; real test has none anyway
        os.link(f.path, D / "images" / "test_v1" / f.name)
        shutil.copy(D / "labels" / "test" / (Path(f.name).stem + ".txt"), D / "labels" / "test_v1")

    cnt = {"train": 0, "val": 0, "test_bed": 0, "dropped": 0}
    for vd in sorted((R / "india_labeled").iterdir()):
        for im in sorted(vd.glob("*.jpg")):
            f = int(im.stem.split("__f")[1]); b = f // BLOCK; pos = f % BLOCK
            kind = "test_bed" if b % 5 == 4 else "val" if b % 5 == 2 else "train"
            if kind == "train" and ((b % 5 in (3, 0) and pos < GAP) or (b % 5 in (1, 3) and pos >= BLOCK - GAP)):
                cnt["dropped"] += 1; continue
            lab = im.with_suffix(".txt").read_text()
            if kind == "train":
                os.symlink(im, D / "images" / "train" / im.name)
            else:
                shutil.copy(im, D / "images" / kind / im.name)
            (D / "labels" / kind / (im.stem + ".txt")).write_text(lab); cnt[kind] += 1
            if kind == "test_bed":
                os.link(D / "images" / kind / im.name, D / "images" / "test_bed_v1" / im.name)
                (D / "labels" / "test_bed_v1" / (im.stem + ".txt")).write_text("".join(l for l in lab.splitlines(True) if int(l.split()[0]) < 100))
    print(cnt)
    names = lambda cs: "".join(f"  {i}: {json.dumps(c)}\n" for i, c in enumerate(cs))
    base = f"path: {D}\ntrain: images/train\nval: images/val\n"
    (D / "data.yaml").write_text(base + f"test: images/test\nnc: {len(C)}\nnames:\n" + names(C))
    (D / "data_bed.yaml").write_text(base + f"test: images/test_bed\nnc: {len(C)}\nnames:\n" + names(C))
    (D / "data_v1.yaml").write_text(base + f"test: images/test_v1\nnc: 100\nnames:\n" + names(C[:100]))
    (D / "data_bed_v1.yaml").write_text(base + f"test: images/test_bed_v1\nnc: 100\nnames:\n" + names(C[:100]))


if __name__ == "__main__":
    main()
