import nibabel as nib
import numpy as np
from scipy.ndimage import label
from pathlib import Path

IN_ROOT = Path("data/tcga_masks")
OUT_ROOT = Path("data/tcga_masks_lcc")
OUT_ROOT.mkdir(exist_ok=True)

for nii in IN_ROOT.rglob("*.nii.gz"):

    img = nib.load(nii)
    mask = img.get_fdata()

    tumor = (mask == 2)

    labeled, num = label(tumor)

    if num == 0:
        continue

    sizes = [(labeled == i).sum() for i in range(1, num + 1)]
    largest = np.argmax(sizes) + 1

    clean = (labeled == largest).astype(np.uint8)

    out_dir = OUT_ROOT / nii.parent.name
    out_dir.mkdir(exist_ok=True)

    nib.save(
        nib.Nifti1Image(clean, img.affine),
        out_dir / nii.name
    )