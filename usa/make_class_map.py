"""Write classes.txt + class_map.yaml for the USA: the 209 handbook rows, id 0..208 (= row - 1).

Template folder templates/NNN_* has NNN = row = id + 1 (extract_templates.py), so every class has a template.
There are no hand-sorted class folders: autobox_us.py names the class of every real sign itself.
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parent


def main():
    classes = {int(k): v for k, v in json.loads((ROOT / "classes.json").read_text()).items()}
    names = [classes[i] for i in range(1, len(classes) + 1)]
    (ROOT / "classes.txt").write_text("\n".join(names) + "\n")
    (ROOT / "class_map.yaml").write_text(yaml.safe_dump(
        {"names": dict(enumerate(names)), "folders": {n: i for i, n in enumerate(names)}, "excluded": []},
        sort_keys=False, allow_unicode=True))
    print(f"{len(names)} classes")


if __name__ == "__main__":
    main()
