import numpy as np
import json
from pathlib import Path

def load_case(case_dir: Path):
    data = np.load(case_dir / "data.npz")
    image = data["image"]   # float16
    mask = data["mask"]     # uint8

    with open(case_dir / "meta.json", "r") as f:
        meta = json.load(f)

    return image, mask, meta
