import numpy as np
from scipy.ndimage import distance_transform_edt


def surface_distances(pred, gt):
    """
    pred, gt: numpy arrays (binary)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 or gt.sum() == 0:
        return None

    pred_border = pred ^ np.logical_and(
        np.logical_and(
            np.roll(pred, 1, axis=0),
            np.roll(pred, -1, axis=0)
        ),
        np.logical_and(
            np.roll(pred, 1, axis=1),
            np.roll(pred, -1, axis=1)
        )
    )

    gt_border = gt ^ np.logical_and(
        np.logical_and(
            np.roll(gt, 1, axis=0),
            np.roll(gt, -1, axis=0)
        ),
        np.logical_and(
            np.roll(gt, 1, axis=1),
            np.roll(gt, -1, axis=1)
        )
    )

    dt_gt = distance_transform_edt(~gt_border)
    dt_pred = distance_transform_edt(~pred_border)

    sds = np.concatenate([
        dt_gt[pred_border],
        dt_pred[gt_border]
    ])

    return sds


def hd95(pred, gt):
    sds = surface_distances(pred, gt)
    if sds is None or len(sds) == 0:
        return np.inf
    return np.percentile(sds, 95)


def asd(pred, gt):
    sds = surface_distances(pred, gt)
    if sds is None or len(sds) == 0:
        return np.inf
    return np.mean(sds)