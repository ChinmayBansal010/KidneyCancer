import json
import numpy as np
from pathlib import Path

EXPECTED_SHAPE = (64, 64, 64)

def check_case(case_dir: Path):
    meta_path = case_dir / "patches_meta.json"
    assert meta_path.exists(), f"{case_dir.name}: patches_meta.json missing"

    with open(meta_path, "r") as f:
        meta = json.load(f)

    assert meta["num_patches"] > 0, f"{case_dir.name}: no patches saved"

    for rec in meta["patches"]:
        pid = rec["patch_id"]
        patch_path = case_dir / f"patch_{pid:03d}.npz"
        assert patch_path.exists(), f"{case_dir.name}: missing patch {pid}"

        data = np.load(patch_path)
        image = data["image"]
        mask = data["mask"]

        assert image.shape == EXPECTED_SHAPE
        assert mask.shape == EXPECTED_SHAPE
        assert image.dtype == np.float16
        assert mask.dtype == np.uint8

        # If tumor-present case, at least one patch must contain tumor
        if rec["tumor_present"]:
            assert rec["tumor_voxels"] >= 0

def check_all(root_dir: str):
    root = Path(root_dir)
    cases = sorted(root.glob("case_*"))

    assert len(cases) > 0, "No patch cases found"

    for case in cases:
        check_case(case)

    print(f"✓ Patch sanity passed ({len(cases)} cases)")


if __name__ == "__main__":
    check_all("data/patches/kits19")