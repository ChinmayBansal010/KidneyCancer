import json
import numpy as np
from pathlib import Path

def check_case(case_dir: Path):
    data_path = case_dir / "tumor_data.npz"
    meta_path = case_dir / "tumor_meta.json"

    assert data_path.exists(), f"{case_dir.name}: tumor_data.npz missing"
    assert meta_path.exists(), f"{case_dir.name}: tumor_meta.json missing"

    data = np.load(data_path)
    image = data["image"]
    mask = data["mask"]

    with open(meta_path, "r") as f:
        meta = json.load(f)

    assert image.shape == mask.shape, f"{case_dir.name}: shape mismatch"

    if meta["tumor_present"]:
        assert (mask == 2).any(), f"{case_dir.name}: tumor flagged but not found"
        bbox = meta["bbox"]
        assert bbox["zmin"] < bbox["zmax"]
        assert bbox["ymin"] < bbox["ymax"]
        assert bbox["xmin"] < bbox["xmax"]
    else:
        assert not (mask == 2).any(), f"{case_dir.name}: tumor present but flagged absent"

def check_all(root_dir: str):
    root = Path(root_dir)
    cases = sorted(root.glob("case_*"))

    assert len(cases) > 0, "No tumor ROI cases found"

    for case in cases:
        check_case(case)

    print(f"✓ Tumor localization sanity passed ({len(cases)} cases)")



if __name__ == "__main__":
    check_all("data/tumor_roi/kits19")