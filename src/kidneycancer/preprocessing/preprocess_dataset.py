"""Dataset preprocessing pipeline for KiTS19 volumes."""

import gc
import json
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from .config import CROP_MARGIN, HU_MAX, HU_MIN, OUT_ROOT, RAW_ROOT, TARGET_SPACING
from .cropping import crop_to_mask
from .loader import load_nifti
from .normalization import hu_window, z_score
from .resampling import resample


def preprocess_case(case_path: Path, out_root: str | Path = OUT_ROOT) -> None:
    """Preprocess one KiTS19 case and save the normalized cropped volume."""
    out_dir = Path(out_root) / case_path.name
    out_dir.mkdir(parents=True, exist_ok=True)

    _, image_itk, meta = load_nifti(case_path / "imaging.nii.gz")
    _, mask_itk, _ = load_nifti(case_path / "segmentation.nii.gz")

    image_itk = resample(image_itk, TARGET_SPACING, is_label=False)
    mask_itk = resample(mask_itk, TARGET_SPACING, is_label=True)

    image = sitk.GetArrayFromImage(image_itk).astype(np.float32, copy=False)
    mask = sitk.GetArrayFromImage(mask_itk).astype(np.uint8, copy=False)

    del image_itk, mask_itk
    gc.collect()

    image = hu_window(image, HU_MIN, HU_MAX)
    image = z_score(image)
    image, mask, crop_meta = crop_to_mask(image, mask, CROP_MARGIN)

    np.savez_compressed(
        out_dir / "data.npz",
        image=image.astype(np.float16, copy=False),
        mask=mask,
    )

    meta.update(crop_meta)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    del image, mask
    gc.collect()


def preprocess_all(
    raw_root: str | Path | None = RAW_ROOT,
    out_root: str | Path | None = OUT_ROOT,
    workers: int | None = None,
) -> None:
    """Preprocess every KiTS19 case found under the configured raw root."""
    raw_root = Path(raw_root or RAW_ROOT)
    out_root = Path(out_root or OUT_ROOT)
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root does not exist: {raw_root}")

    cases = sorted(raw_root.glob("case_*"))
    if not cases:
        raise FileNotFoundError(f"No KiTS19 case directories found in: {raw_root}")

    max_workers = workers or max(1, min(4, os.cpu_count() // 2))

    with Pool(processes=max_workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(
                    partial(preprocess_case, out_root=out_root),
                    cases,
                ),
                total=len(cases),
                desc="Preprocessing KiTS19",
            )
        )
