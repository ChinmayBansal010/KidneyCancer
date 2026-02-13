TUMOR_ROI_ROOT = "data/tumor_roi/kits19"
OUT_ROOT = "data/patches/kits19"

PATCH_SIZE = (64, 64, 64)   # (Z, Y, X)

# Fixed deterministic offsets (voxels)
OFFSETS = [
    (0, 0, 0),
    (8, 0, 0),
    (-8, 0, 0),
    (0, 8, 0),
    (0, -8, 0),
]
