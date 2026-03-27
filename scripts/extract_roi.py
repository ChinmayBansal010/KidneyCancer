import numpy as np
import nibabel as nib
import json
from pathlib import Path

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from kidneycancer.utils.logging_utils import configure_logging

CT_ROOT = Path("data/tcga_nifti")
MASK_ROOT = Path("data/tcga_masks_lcc")
OUT_ROOT = Path("data/tcga_roi")

MZ, MY, MX = 32, 64, 64
LOGGER = configure_logging("kidneycancer.extract_roi")


def main() -> None:
    """Extract fixed-size TCGA ROIs around predicted tumor masks."""
    for ct in CT_ROOT.rglob("*.nii.gz"):
        subtype = ct.parts[-3]
        pid = ct.stem.replace(".nii", "")

        mask_file = MASK_ROOT / ct.parent.parent.name / ct.name
        if not mask_file.exists():
            continue

        ct_vol = nib.load(ct).get_fdata()
        mask = nib.load(mask_file).get_fdata() > 0

        coords = np.argwhere(mask)
        if len(coords) == 0:
            LOGGER.warning("Skipping %s because the mask is empty", pid)
            continue

        zc, yc, xc = coords.mean(axis=0)
        zc, yc, xc = int(zc), int(yc), int(xc)

        z0, z1 = max(0, zc - MZ), min(ct_vol.shape[0], zc + MZ)
        y0, y1 = max(0, yc - MY), min(ct_vol.shape[1], yc + MY)
        x0, x1 = max(0, xc - MX), min(ct_vol.shape[2], xc + MX)

        roi = ct_vol[z0:z1, y0:y1, x0:x1]
        roi_mask = mask[z0:z1, y0:y1, x0:x1]

        out = OUT_ROOT / subtype / pid
        out.mkdir(parents=True, exist_ok=True)

        np.save(out / "ct.npy", roi.astype(np.float32))
        np.save(out / "mask.npy", roi_mask.astype(np.uint8))

        meta = {
            "subtype": subtype,
            "center": [zc, yc, xc],
            "bbox": [z0, z1, y0, y1, x0, x1],
            "roi_shape": list(map(int, roi.shape)),
        }

        with open(out / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
        LOGGER.info("Saved ROI for %s/%s", subtype, pid)


if __name__ == "__main__":
    main()
