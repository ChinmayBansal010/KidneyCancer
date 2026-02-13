import numpy as np

def compute_kidney_bbox(mask, kidney_labels, margin):
    kidney_mask = np.isin(mask, kidney_labels)

    if not kidney_mask.any():
        raise RuntimeError("No kidney voxels found in mask")

    coords = np.where(kidney_mask)

    zmin, ymin, xmin = coords[0].min(), coords[1].min(), coords[2].min()
    zmax, ymax, xmax = coords[0].max(), coords[1].max(), coords[2].max()

    zmin = max(0, zmin - margin)
    ymin = max(0, ymin - margin)
    xmin = max(0, xmin - margin)

    zmax = min(mask.shape[0], zmax + margin)
    ymax = min(mask.shape[1], ymax + margin)
    xmax = min(mask.shape[2], xmax + margin)

    return (zmin, zmax, ymin, ymax, xmin, xmax)
