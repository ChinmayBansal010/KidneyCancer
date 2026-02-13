import numpy as np
import json
from pathlib import Path

def load_case(case_dir: Path):
    data = np.load(case_dir / "kidney_data.npz")
    image = data["image"]
    mask = data["mask"]

    with open(case_dir / "kidney_meta.json", "r") as f:
        meta = json.load(f)

    return image, mask, meta
