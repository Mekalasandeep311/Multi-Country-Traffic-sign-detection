"""Write classes.txt + class_map.yaml for Japan: the 120 Merged folders, id 0..119 in classes.json order.

Template folder templates/NNN_* has NNN = id + 1 (extract_templates.py), so every class has a template.
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
DATA = ROOT / "classified Data japan" / "Merged"
# folder whose real frames show another class's sign (checked visually, user decision 2026-10-14):
# the "Rest Area" folder holds P + food + fuel signs = "Rest Area - Parking - Fuel Stand"
FOLDER_TO_CLASS = {"Rest Area": "Rest Area - Parking - Fuel Stand"}


def main():
    classes = {int(k): v for k, v in json.loads((ROOT / "classes.json").read_text()).items()}
    names = [classes[i] for i in range(1, len(classes) + 1)]
    folders = sorted(p.name for p in DATA.iterdir() if p.is_dir())
    assert set(folders) == set(names), set(folders) ^ set(names)   # order = template ids (classes.json)
    (ROOT / "classes.txt").write_text("\n".join(names) + "\n")
    (ROOT / "class_map.yaml").write_text(yaml.safe_dump(
        {"names": dict(enumerate(names)), "folders": {f: names.index(FOLDER_TO_CLASS.get(f, f)) for f in names}, "excluded": []},
        sort_keys=False, allow_unicode=True))
    print(f"{len(names)} classes, {len(folders)} folders mapped")


if __name__ == "__main__":
    main()
