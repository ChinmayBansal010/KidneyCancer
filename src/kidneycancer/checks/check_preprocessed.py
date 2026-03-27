"""Sanity checks for preprocessed KiTS19 volumes."""

from pathlib import Path

import logging
import numpy as np


ALLOWED_LABELS = {0, 1, 2}
LOGGER = logging.getLogger(__name__)


def check_case(case_dir: Path) -> None:
    """Validate one preprocessed case directory."""
    data_path = case_dir / "data.npz"
    meta_path = case_dir / "meta.json"

    assert data_path.exists(), f"{case_dir.name}: data.npz missing"
    assert meta_path.exists(), f"{case_dir.name}: meta.json missing"

    data = np.load(data_path)
    image = data["image"]
    mask = data["mask"]

    assert image.shape == mask.shape, f"{case_dir.name}: shape mismatch"
    assert image.dtype == np.float16, f"{case_dir.name}: image not float16"
    assert mask.dtype == np.uint8, f"{case_dir.name}: mask not uint8"
    assert np.isfinite(image).all(), f"{case_dir.name}: NaN or inf in image"

    labels = set(np.unique(mask).tolist())
    assert labels.issubset(ALLOWED_LABELS), f"{case_dir.name}: invalid labels {labels}"
    assert image.size > 0, f"{case_dir.name}: empty image"


def check_all(root_dir: str) -> None:
    """Run validation checks across all preprocessed cases."""
    root = Path(root_dir)
    cases = sorted(root.glob("case_*"))
    assert cases, "No preprocessed cases found"

    for case in cases:
        check_case(case)

    LOGGER.info("Preprocessed sanity passed (%s cases)", len(cases))


if __name__ == "__main__":
    check_all("data/preprocessed/kits19")
