import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc

from .config import PREPROCESSED_ROOT, OUT_ROOT, KIDNEY_LABELS, BBOX_MARGIN
from .loader import load_case
from .bbox import compute_kidney_bbox

def localize_case(case_dir: Path, out_root: str | Path = OUT_ROOT):
    image, mask, meta = load_case(case_dir)

    zmin, zmax, ymin, ymax, xmin, xmax = compute_kidney_bbox(
        mask, KIDNEY_LABELS, BBOX_MARGIN
    )

    kidney_image = image[zmin:zmax, ymin:ymax, xmin:xmax]
    kidney_mask = mask[zmin:zmax, ymin:ymax, xmin:xmax]

    out_dir = Path(out_root) / case_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "kidney_data.npz",
        image=kidney_image,
        mask=kidney_mask
    )

    kidney_meta = {
        "bbox": {
            "zmin": int(zmin), "zmax": int(zmax),
            "ymin": int(ymin), "ymax": int(ymax),
            "xmin": int(xmin), "xmax": int(xmax)
        },
        "original_shape": list(image.shape),
        "kidney_shape": list(kidney_image.shape),
        "source_case": case_dir.name
    }

    with open(out_dir / "kidney_meta.json", "w") as f:
        json.dump(kidney_meta, f, indent=2)

    del image, mask, kidney_image, kidney_mask
    gc.collect()

def localize_all(
    preprocessed_root: str | Path | None = PREPROCESSED_ROOT,
    out_root: str | Path | None = OUT_ROOT,
):
    root = Path(preprocessed_root or PREPROCESSED_ROOT)
    out_root = Path(out_root or OUT_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"Preprocessed root does not exist: {root}")
    cases = sorted(root.glob("case_*"))
    if not cases:
        raise FileNotFoundError(f"No preprocessed case directories found in: {root}")

    for case in tqdm(cases, desc="Kidney localization"):
        localize_case(case, out_root=out_root)
