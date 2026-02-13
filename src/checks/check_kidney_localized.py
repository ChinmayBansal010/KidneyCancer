import json
import numpy as np
from pathlib import Path

def check_case(case_dir: Path):
    data_path = case_dir / "kidney_data.npz"
    meta_path = case_dir / "kidney_meta.json"

    assert data_path.exists(), f"{case_dir.name}: kidney_data.npz missing"
    assert meta_path.exists(), f"{case_dir.name}: kidney_meta.json missing"

    data = np.load(data_path)
    image = data["image"]
    mask = data["mask"]

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # ---- shape ----
    assert image.shape == mask.shape, f"{case_dir.name}: shape mismatch"

    # ---- smaller than original ----
    original_shape = tuple(meta["original_shape"])
    assert all(
        image.shape[i] <= original_shape[i] for i in range(3)
    ), f"{case_dir.name}: ROI larger than original"

    # ---- kidney presence ----
    assert (mask > 0).any(), f"{case_dir.name}: no kidney voxels in ROI"

    # ---- tumor preservation check ----
    if 2 in np.unique(mask):
        assert (mask == 2).sum() > 0, f"{case_dir.name}: tumor vanished"

    # ---- bbox validity ----
    bbox = meta["bbox"]
    assert bbox["zmin"] < bbox["zmax"]
    assert bbox["ymin"] < bbox["ymax"]
    assert bbox["xmin"] < bbox["xmax"]

def check_all(root_dir: str):
    root = Path(root_dir)
    cases = sorted(root.glob("case_*"))

    assert len(cases) > 0, "No kidney ROI cases found"

    for case in cases:
        check_case(case)

    print(f"✓ Kidney localization sanity passed ({len(cases)} cases)")

if __name__ == "__main__":
    check_all("data/kidney_roi/kits19")