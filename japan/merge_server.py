"""Rebuild CDJ/Merged on the server from Classified_Japan _1.._6.

- copies non-empty files per class, dedupes by md5 inside each class
- Classified_Japan_2 is 0-byte: restore those files by name from Japan/Classify_Split
- writes merge_report.csv + merge_lost.txt, then verifies md5 coverage
Args: none = merge + verify;  "verify" = verify only;  "delete" = verify then delete the 6 folders
"""
import csv, hashlib, os, shutil, sys
from collections import defaultdict
from pathlib import Path

CDJ = Path("/data/common_datasets/ai24mtech02003/sandeep_japan/classified Data japan")
MERGED = CDJ / "Merged"
SOURCES = ["Classified_Japan _1", "Classified_Japan_2", "Classified_Japan _3",
           "Classified_Japan _4", "Classified_Japan _5", "Classified_Japan_6"]
RAW = CDJ / "Japan" / "Classify_Split"
RENAME = {"Rest Area ": "Rest Area"}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def resolve(src, raw):
    """Real file for a source entry (0-byte -> raw frame with same name)."""
    if src.stat().st_size > 0:
        return src, False
    r = raw.get(src.name)
    return (r, True) if r is not None and r.stat().st_size > 0 else (None, True)


def scan_sources(raw):
    """class -> list of (resolved path or None, source path, recovered flag)."""
    out = defaultdict(list)
    for s in SOURCES:
        for cdir in sorted((CDJ / s).iterdir()):
            if not cdir.is_dir():
                continue
            cls = RENAME.get(cdir.name, cdir.name)
            for f in sorted(cdir.iterdir()):
                if f.is_file():
                    out[cls].append((*resolve(f, raw), f))
    return out


def raw_index():
    return {p.name: p for p in RAW.rglob("*") if p.is_file()}


def merge():
    for old, new in RENAME.items():
        if (MERGED / old).is_dir() and not (MERGED / new).exists():
            (MERGED / old).rename(MERGED / new)
    raw = raw_index()
    src = scan_sources(raw)
    rows, lost = [], []
    classes = sorted(set(os.listdir(MERGED)) | set(src))
    for cls in classes:
        dst = MERGED / cls
        dst.mkdir(exist_ok=True)
        seen = {md5(p): p.name for p in dst.iterdir() if p.is_file()}
        n_src = n_dup = n_rec = n_lost = n_new = 0
        for real, rec, orig in src.get(cls, []):
            n_src += 1
            if real is None:
                n_lost += 1
                lost.append(f"{cls}/{orig.name}\t{orig.parent.parent.name}")
                continue
            h = md5(real)
            if h in seen:
                n_dup += 1
                continue
            name, k = orig.name, 1
            while (dst / name).exists():
                name = f"{orig.stem}_{k}{orig.suffix}"
                k += 1
            shutil.copy2(real, dst / name)
            seen[h] = name
            n_new += 1
            n_rec += rec
        rows.append([cls, n_src, n_dup, n_rec, n_lost, len(seen)])
        print(f"{len(seen):5d}  {cls}", flush=True)
    with open(CDJ / "merge_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "source_files", "duplicates_dropped", "recovered_from_raw",
                    "lost_0byte", "merged_images"])
        w.writerows(rows)
    (CDJ / "merge_lost.txt").write_text("\n".join(lost) + "\n")
    print("classes", len(rows), "merged images", sum(r[-1] for r in rows), "lost", len(lost))


def verify():
    raw = raw_index()
    src = scan_sources(raw)
    missing = 0
    for cls, items in src.items():
        have = {md5(p) for p in (MERGED / cls).iterdir() if p.is_file()}
        for real, _, orig in items:
            if real is not None and md5(real) not in have:
                missing += 1
                print("MISSING", cls, orig)
    zero = [p for p in MERGED.rglob("*") if p.is_file() and p.stat().st_size == 0]
    print("verify: missing", missing, "zero-byte in Merged", len(zero))
    return missing == 0 and not zero


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "merge"
    if mode == "merge":
        merge()
        ok = verify()
    elif mode == "verify":
        verify()
    elif mode == "delete":
        if not verify():
            sys.exit("verification failed - nothing deleted")
        for s in SOURCES:
            shutil.rmtree(CDJ / s)
            print("deleted", s)
