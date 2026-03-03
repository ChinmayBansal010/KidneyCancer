import nibabel as nib
from pathlib import Path

ROOT = Path("data/tcga_nifti")

print("Checking one file first...\n")

# ---- Check only first file ----
first = next(ROOT.rglob("*.nii.gz"))
img = nib.load(first)

print("File:", first.name)
print("Shape:", img.shape)

print("\nNow counting all shapes...\n")

# ---- Fast shape scan (header only) ----
shapes = {}

for nii in ROOT.rglob("*.nii.gz"):
    try:
        img = nib.load(nii)
        shape = img.shape  # <-- FAST (header only)
        shapes[shape] = shapes.get(shape, 0) + 1
    except Exception as e:
        print(f"[ERROR] {nii.name}: {e}")

print("\n==== UNIQUE SHAPES ====")
for shape, count in shapes.items():
    print(f"{shape}  -->  {count} scans")