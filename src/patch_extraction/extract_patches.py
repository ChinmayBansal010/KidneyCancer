import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc

from .config import TUMOR_ROI_ROOT, OUT_ROOT, PATCH_SIZE, OFFSETS
from .loader import load_case
from .sampler import get_center, extract_patch

def extract_case(case_dir: Path):
    image, mask, meta = load_case(case_dir)

    tumor_present = meta["tumor_present"]
    center = get_center(mask, tumor_present)

    out_dir = Path(OUT_ROOT) / case_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_records = []
    patch_id = 0

    for dz, dy, dx in OFFSETS:
        c = (center[0] + dz, center[1] + dy, center[2] + dx)
        patch_img, patch_mask = extract_patch(image, mask, c, PATCH_SIZE)

        if patch_img is None:
            continue

        np.savez_compressed(
            out_dir / f"patch_{patch_id:03d}.npz",
            image=patch_img.astype(np.float16),
            mask=patch_mask.astype(np.uint8)
        )

        patch_records.append({
            "patch_id": patch_id,
            "center": c,
            "tumor_present": tumor_present,
            "tumor_voxels": int((patch_mask == 2).sum())
        })

        patch_id += 1

    with open(out_dir / "patches_meta.json", "w") as f:
        json.dump({
            "source_case": case_dir.name,
            "num_patches": len(patch_records),
            "patch_size": PATCH_SIZE,
            "patches": patch_records
        }, f, indent=2)

    del image, mask
    gc.collect()

def extract_all():
    root = Path(TUMOR_ROI_ROOT)
    cases = sorted(root.glob("case_*"))

    for case in tqdm(cases, desc="Patch extraction"):
        extract_case(case)
