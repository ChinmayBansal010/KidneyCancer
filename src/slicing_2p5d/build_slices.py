import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .config import TUMOR_ROI_ROOT, OUT_ROOT, SLICE_SIZE, TUMOR_LABEL
from .slicer import (
    find_largest_slice,
    extract_slice,
    center_crop_or_pad
)

def process_case(case_dir: Path):
    data = np.load(case_dir / "tumor_data.npz")
    image = data["image"]
    mask = data["mask"]

    tumor_present = bool((mask == TUMOR_LABEL).any())

    if tumor_present:
        z_idx = find_largest_slice(mask, axis=0, tumor_label=TUMOR_LABEL)
        y_idx = find_largest_slice(mask, axis=1, tumor_label=TUMOR_LABEL)
        x_idx = find_largest_slice(mask, axis=2, tumor_label=TUMOR_LABEL)
    else:
        # fallback to center
        z_idx = image.shape[0] // 2
        y_idx = image.shape[1] // 2
        x_idx = image.shape[2] // 2

    axial_img = extract_slice(image, z_idx, axis=0)
    coronal_img = extract_slice(image, y_idx, axis=1)
    sagittal_img = extract_slice(image, x_idx, axis=2)

    axial_m = extract_slice(mask, z_idx, axis=0)
    coronal_m = extract_slice(mask, y_idx, axis=1)
    sagittal_m = extract_slice(mask, x_idx, axis=2)

    axial_img = center_crop_or_pad(axial_img, SLICE_SIZE)
    coronal_img = center_crop_or_pad(coronal_img, SLICE_SIZE)
    sagittal_img = center_crop_or_pad(sagittal_img, SLICE_SIZE)

    axial_m = center_crop_or_pad(axial_m, SLICE_SIZE)
    coronal_m = center_crop_or_pad(coronal_m, SLICE_SIZE)
    sagittal_m = center_crop_or_pad(sagittal_m, SLICE_SIZE)

    slice_2p5d = np.stack(
        [axial_img, coronal_img, sagittal_img],
        axis=0
    )

    mask_2p5d = np.stack(
        [axial_m, coronal_m, sagittal_m],
        axis=0
    )

    out_dir = Path(OUT_ROOT) / case_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "slice_2p5d.npz",
        image=slice_2p5d.astype(np.float32),
        mask=mask_2p5d.astype(np.uint8)
    )

    with open(out_dir / "slice_meta.json", "w") as f:
        json.dump({
            "tumor_present": tumor_present,
            "indices": {
                "axial": int(z_idx),
                "coronal": int(y_idx),
                "sagittal": int(x_idx)
            },
            "slice_size": SLICE_SIZE,
            "source_case": case_dir.name
        }, f, indent=2)


def build_all():
    root = Path(TUMOR_ROI_ROOT)
    cases = sorted(root.glob("case_*"))

    for case in tqdm(cases, desc="2.5D slicing"):
        process_case(case)
