import numpy as np

def crop_to_mask(image, mask, margin):
    coords = np.where(mask > 0)

    zmin, ymin, xmin = coords[0].min(), coords[1].min(), coords[2].min()
    zmax, ymax, xmax = coords[0].max(), coords[1].max(), coords[2].max()

    zmin = max(0, zmin - margin)
    ymin = max(0, ymin - margin)
    xmin = max(0, xmin - margin)

    zmax = min(mask.shape[0], zmax + margin)
    ymax = min(mask.shape[1], ymax + margin)
    xmax = min(mask.shape[2], xmax + margin)

    cropped_img = image[zmin:zmax, ymin:ymax, xmin:xmax]
    cropped_mask = mask[zmin:zmax, ymin:ymax, xmin:xmax]

    crop_meta = {
        "zmin": int(zmin),
        "ymin": int(ymin),
        "xmin": int(xmin),
        "shape": cropped_img.shape
    }

    return cropped_img, cropped_mask, crop_meta
