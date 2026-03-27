import numpy as np

def get_center(mask, tumor_present):
    if tumor_present:
        coords = np.where(mask == 2)
    else:
        coords = np.where(mask > 0)

    z = int(coords[0].mean())
    y = int(coords[1].mean())
    x = int(coords[2].mean())

    return (z, y, x)

def pad_if_needed(image, mask, patch_size):
    dz, dy, dx = patch_size
    iz, iy, ix = image.shape

    pad_z = max(0, dz - iz)
    pad_y = max(0, dy - iy)
    pad_x = max(0, dx - ix)

    pad_before = (
        pad_z // 2,
        pad_y // 2,
        pad_x // 2
    )
    pad_after = (
        pad_z - pad_before[0],
        pad_y - pad_before[1],
        pad_x - pad_before[2]
    )

    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        image = np.pad(
            image,
            (
                (pad_before[0], pad_after[0]),
                (pad_before[1], pad_after[1]),
                (pad_before[2], pad_after[2])
            ),
            mode="constant",
            constant_values=0.0
        )

        mask = np.pad(
            mask,
            (
                (pad_before[0], pad_after[0]),
                (pad_before[1], pad_after[1]),
                (pad_before[2], pad_after[2])
            ),
            mode="constant",
            constant_values=0
        )

    return image, mask

def extract_patch(image, mask, center, patch_size):
    # Ensure minimum size
    image, mask = pad_if_needed(image, mask, patch_size)

    dz, dy, dx = patch_size
    zc, yc, xc = center

    zmin = zc - dz // 2
    ymin = yc - dy // 2
    xmin = xc - dx // 2

    zmax = zmin + dz
    ymax = ymin + dy
    xmax = xmin + dx

    # Clamp center if needed after padding
    zmin = max(0, min(zmin, image.shape[0] - dz))
    ymin = max(0, min(ymin, image.shape[1] - dy))
    xmin = max(0, min(xmin, image.shape[2] - dx))

    zmax = zmin + dz
    ymax = ymin + dy
    xmax = xmin + dx

    return (
        image[zmin:zmax, ymin:ymax, xmin:xmax],
        mask[zmin:zmax, ymin:ymax, xmin:xmax]
    )
