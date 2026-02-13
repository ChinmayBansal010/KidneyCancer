import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import SimpleITK as sitk
import gc
import os

from .config import (
    HU_MIN, HU_MAX,
    TARGET_SPACING,
    RAW_ROOT, OUT_ROOT,
    CROP_MARGIN
)
from .loader import load_nifti
from .resampling import resample
from .normalization import hu_window, z_score
from .cropping import crop_to_mask

def preprocess_case(case_path: Path):
    out_dir = Path(OUT_ROOT) / case_path.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- LOAD ----
    _, img_itk, meta = load_nifti(case_path / "imaging.nii.gz")
    _, mask_itk, _ = load_nifti(case_path / "segmentation.nii.gz")

    # ---- RESAMPLE ----
    img_itk = resample(img_itk, TARGET_SPACING, is_label=False)
    mask_itk = resample(mask_itk, TARGET_SPACING, is_label=True)

    img = sitk.GetArrayFromImage(img_itk).astype(np.float32, copy=False)
    mask = sitk.GetArrayFromImage(mask_itk).astype(np.uint8, copy=False)

    # Explicitly free ITK objects early
    del img_itk, mask_itk
    gc.collect()

    # ---- NORMALIZE ----
    img = hu_window(img, HU_MIN, HU_MAX)
    img = z_score(img)

    # ---- CROP ----
    img, mask, crop_meta = crop_to_mask(img, mask, CROP_MARGIN)

    # ---- SAVE ----
    np.savez_compressed(
        out_dir / "data.npz",
        image=img.astype(np.float16, copy=False),
        mask=mask
    )

    meta.update(crop_meta)

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ---- CLEANUP ----
    del img, mask
    gc.collect()

def preprocess_all():
    raw_root = Path(RAW_ROOT)
    cases = sorted(raw_root.glob("case_*"))

    # Windows-safe worker count
    max_workers = max(1, min(4, os.cpu_count() // 2))

    with Pool(processes=max_workers) as pool:
        list(tqdm(
            pool.imap_unordered(preprocess_case, cases),
            total=len(cases),
            desc="Preprocessing KiTS19"
        ))
