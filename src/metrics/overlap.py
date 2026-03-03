import torch


def dice_score(pred, gt, eps=1e-5):
    """
    pred, gt: torch tensors [Z,Y,X] or [B,Z,Y,X]
    """
    pred = (pred > 0).float()
    gt = (gt > 0).float()

    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum()

    return (2.0 * intersection + eps) / (union + eps)


def iou_score(pred, gt, eps=1e-5):
    pred = (pred > 0).float()
    gt = (gt > 0).float()

    intersection = (pred * gt).sum()
    union = ((pred + gt) > 0).float().sum()

    return (intersection + eps) / (union + eps)