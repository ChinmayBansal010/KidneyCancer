import json
import numpy as np
from pathlib import Path
import logging

LOGGER = logging.getLogger(__name__)

EXPECTED_SHAPE = (3, 128, 128)

def check_case(case_dir: Path):
    data_path = case_dir / "slice_2p5d.npz"
    meta_path = case_dir / "slice_meta.json"

    assert data_path.exists(), f"{case_dir.name}: missing slice data"
    assert meta_path.exists(), f"{case_dir.name}: missing meta"

    data = np.load(data_path)
    img = data["image"]
    mask = data["mask"]

    assert img.shape == EXPECTED_SHAPE
    assert mask.shape == EXPECTED_SHAPE
    assert img.dtype == np.float32
    assert mask.dtype == np.uint8

    with open(meta_path) as f:
        meta = json.load(f)

    if meta["tumor_present"]:
        assert (mask == 2).any(), f"{case_dir.name}: tumor flagged but not found"

def check_all(root_dir):
    root = Path(root_dir)
    cases = sorted(root.glob("case_*"))
    assert len(cases) > 0

    for case in cases:
        check_case(case)

    LOGGER.info("2.5D slicing sanity passed (%s cases)", len(cases))


if __name__ == "__main__":
    check_all("data/slices_2p5d/kits19")
