"""templates/<NNN>_<class>/handbook.png from the rule-book pictures (NNN = class id)."""
import json, shutil
from pathlib import Path
R = Path(__file__).parent
cls = json.load(open(R / "class_map.json"))["classes"]
pics = sorted((R / "handbook" / "pictures").iterdir())
assert len(pics) == 98
for i, p in enumerate(pics):
    d = R / "templates" / f"{i:03d}_{cls[i].replace('/', '-')}"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(p, d / ("handbook" + p.suffix.lower()))
print("templates", len(list((R / "templates").iterdir())))
