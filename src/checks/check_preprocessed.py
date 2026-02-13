import numpy as np
from pathlib import Path

ALLOWED_LABELS = {0, 1, 2}

def check_case(case_dir: Path):
    data_path = case_dir / "data.npz"
    meta_path = case_dir / "meta.json"

    assert data_path.exists(), f"{case_dir.name}: data.npz missing"
    assert meta_path.exists(), f"{case_dir.name}: meta.json missing"

    data = np.load(data_path)
    image = data["image"]
    mask = data["mask"]

    # ---- shape ----
    assert image.shape == mask.shape, f"{case_dir.name}: shape mismatch"

    # ---- dtype ----
    assert image.dtype == np.float16, f"{case_dir.name}: image not float16"
    assert mask.dtype == np.uint8, f"{case_dir.name}: mask not uint8"

    # ---- numerical sanity ----
    assert np.isfinite(image).all(), f"{case_dir.name}: NaN or inf in image"

    # ---- mask labels ----
    labels = set(np.unique(mask).tolist())
    assert labels.issubset(ALLOWED_LABELS), (
        f"{case_dir.name}: invalid labels {labels}"
    )

    # ---- non-empty ----
    assert image.size > 0, f"{case_dir.name}: empty image"

def check_all(root_dir: str):
    root = Path(root_dir)
    cases = sorted(root.glob("case_*"))

    assert len(cases) > 0, "No preprocessed cases found"

    for case in cases:
        check_case(case)

    print(f"✓ Preprocessed sanity passed ({len(cases)} cases)")

if __name__ == "__main__":
    check_all("data/preprocessed/kits19")