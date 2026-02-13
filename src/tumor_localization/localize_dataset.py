import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc

from .config import KIDNEY_ROI_ROOT, OUT_ROOT, TUMOR_LABEL, BBOX_MARGIN
from .loader import load_case
from .bbox import compute_tumor_bbox

def localize_case(case_dir: Path):
    image, mask, meta = load_case(case_dir)

    bbox = compute_tumor_bbox(mask, TUMOR_LABEL, BBOX_MARGIN)

    out_dir = Path(OUT_ROOT) / case_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if bbox is None:
        # Tumor-absent case
        np.savez_compressed(
            out_dir / "tumor_data.npz",
            image=image,
            mask=mask
        )

        tumor_meta = {
            "tumor_present": False,
            "source_case": case_dir.name,
            "kidney_shape": list(image.shape)
        }
    else:
        zmin, zmax, ymin, ymax, xmin, xmax = bbox

        tumor_image = image[zmin:zmax, ymin:ymax, xmin:xmax]
        tumor_mask = mask[zmin:zmax, ymin:ymax, xmin:xmax]

        np.savez_compressed(
            out_dir / "tumor_data.npz",
            image=tumor_image,
            mask=tumor_mask
        )

        tumor_meta = {
            "tumor_present": True,
            "bbox": {
                "zmin": int(zmin), "zmax": int(zmax),
                "ymin": int(ymin), "ymax": int(ymax),
                "xmin": int(xmin), "xmax": int(xmax)
            },
            "tumor_shape": list(tumor_image.shape),
            "source_case": case_dir.name
        }

        del tumor_image, tumor_mask

    with open(out_dir / "tumor_meta.json", "w") as f:
        json.dump(tumor_meta, f, indent=2)

    del image, mask
    gc.collect()

def localize_all():
    root = Path(KIDNEY_ROI_ROOT)
    cases = sorted(root.glob("case_*"))

    for case in tqdm(cases, desc="Tumor localization"):
        localize_case(case)
