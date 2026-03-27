import numpy as np

def find_largest_slice(mask, axis, tumor_label):
    """
    Returns index of slice with max tumor area along axis
    """
    areas = []

    for i in range(mask.shape[axis]):
        if axis == 0:
            slice_ = mask[i, :, :]
        elif axis == 1:
            slice_ = mask[:, i, :]
        else:
            slice_ = mask[:, :, i]

        areas.append((slice_ == tumor_label).sum())

    return int(np.argmax(areas))


def extract_slice(volume, index, axis):
    if axis == 0:
        return volume[index, :, :]
    elif axis == 1:
        return volume[:, index, :]
    else:
        return volume[:, :, index]


def center_crop_or_pad(slice_, out_size):
    h, w = slice_.shape
    out = np.zeros((out_size, out_size), dtype=slice_.dtype)

    cy, cx = h // 2, w // 2
    oy, ox = out_size // 2, out_size // 2

    y1 = max(0, cy - oy)
    y2 = min(h, cy + oy)
    x1 = max(0, cx - ox)
    x2 = min(w, cx + ox)

    out_y1 = oy - (cy - y1)
    out_y2 = out_y1 + (y2 - y1)
    out_x1 = ox - (cx - x1)
    out_x2 = out_x1 + (x2 - x1)

    out[out_y1:out_y2, out_x1:out_x2] = slice_[y1:y2, x1:x2]
    return out
