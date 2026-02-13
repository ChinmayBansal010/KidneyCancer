import numpy as np

def compute_tumor_bbox(mask, tumor_label, margin):
    tumor_mask = mask == tumor_label

    if not tumor_mask.any():
        return None  # tumor-absent case

    coords = np.where(tumor_mask)

    zmin, ymin, xmin = coords[0].min(), coords[1].min(), coords[2].min()
    zmax, ymax, xmax = coords[0].max(), coords[1].max(), coords[2].max()

    zmin = max(0, zmin - margin)
    ymin = max(0, ymin - margin)
    xmin = max(0, xmin - margin)

    zmax = min(mask.shape[0], zmax + margin)
    ymax = min(mask.shape[1], ymax + margin)
    xmax = min(mask.shape[2], xmax + margin)

    return (zmin, zmax, ymin, ymax, xmin, xmax)
