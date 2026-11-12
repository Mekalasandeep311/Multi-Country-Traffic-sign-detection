"""Map the real dataset's folder names to class ids and write class_map.yaml + classes.txt.

ids 0..155   = handbook classes 1..156 (YOLO id = handbook number - 1)
ids 156..    = real-data folders with no handbook equivalent
Folders listed in EXCLUDE are not a single sign type and are skipped.
"""
import difflib
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
DATA = ROOT / "Traffic Signs Singapore"

OVERRIDES = {  # folder -> handbook number, where names differ
    "Advance Directional Sign (non-expressway)": 2,
    "Confirmation Directional Sign (non-expressway)": 14,
}
NEW_CLASSES = ["Lane Shift Right", "No Parking", "No riding of pedal-cycles"]
EXCLUDE = ["Road Signs", "No U-Turn", "Shared Track"]  # mixed scenes / Boston-only, no SG template (user removed)


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def main():
    handbook = {int(k): v for k, v in json.loads((ROOT / "classes.json").read_text()).items()}
    handbook.setdefault(85, "Right Turn (for specific vehicles)")
    by_norm = {norm(v): k for k, v in handbook.items()}

    names = [handbook[i] for i in range(1, 157)] + NEW_CLASSES
    folders = {}
    for d in sorted(p.name for p in DATA.iterdir() if p.is_dir()):
        if d in EXCLUDE or d.startswith(("_", "dataset")):
            continue
        if d in NEW_CLASSES:
            folders[d] = 156 + NEW_CLASSES.index(d)
        elif d in OVERRIDES:
            folders[d] = OVERRIDES[d] - 1
        elif norm(d) in by_norm:
            folders[d] = by_norm[norm(d)] - 1
        else:
            guess = difflib.get_close_matches(norm(d), list(by_norm), n=1)
            raise SystemExit(f"unmapped folder {d!r} (closest: {guess}) - add it to OVERRIDES")

    (ROOT / "classes.txt").write_text("\n".join(names) + "\n")
    (ROOT / "class_map.yaml").write_text(yaml.safe_dump(
        {"names": dict(enumerate(names)), "folders": folders, "excluded": EXCLUDE},
        sort_keys=False, allow_unicode=True))
    print(f"{len(names)} classes, {len(folders)} real folders mapped, excluded {EXCLUDE}")
    for d, i in folders.items():
        if norm(d) != norm(names[i]):
            print(f"  {d!r} -> {i}: {names[i]}")


if __name__ == "__main__":
    main()
