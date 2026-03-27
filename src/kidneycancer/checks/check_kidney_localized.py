"""Sanity checks for kidney-localized KiTS19 volumes."""

import json
from pathlib import Path

import logging
import numpy as np

LOGGER = logging.getLogger(__name__)

def check_case(case_dir: Path) -> None:
    """Validate one kidney-localized case directory."""
    data_path = case_dir / "kidney_data.npz"
    meta_path = case_dir / "kidney_meta.json"

    assert data_path.exists(), f"{case_dir.name}: kidney_data.npz missing"
    assert meta_path.exists(), f"{case_dir.name}: kidney_meta.json missing"

    data = np.load(data_path)
    image = data["image"]
    mask = data["mask"]

    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    assert image.shape == mask.shape, f"{case_dir.name}: shape mismatch"

    original_shape = tuple(meta["original_shape"])
    assert all(image.shape[i] <= original_shape[i] for i in range(3)), (
        f"{case_dir.name}: ROI larger than original"
    )
    assert (mask > 0).any(), f"{case_dir.name}: no kidney voxels in ROI"

    if 2 in np.unique(mask):
        assert (mask == 2).sum() > 0, f"{case_dir.name}: tumor vanished"

    bbox = meta["bbox"]
    assert bbox["zmin"] < bbox["zmax"]
    assert bbox["ymin"] < bbox["ymax"]
    assert bbox["xmin"] < bbox["xmax"]


def check_all(root_dir: str) -> None:
    """Run validation checks across all kidney-localized cases."""
    root = Path(root_dir)
    cases = sorted(root.glob("case_*"))
    assert cases, "No kidney ROI cases found"

    for case in cases:
        check_case(case)

    LOGGER.info("Kidney localization sanity passed (%s cases)", len(cases))


if __name__ == "__main__":
    check_all("data/kidney_roi/kits19")
