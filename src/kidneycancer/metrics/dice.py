import torch

@torch.no_grad()
def dice_score(pred, target, class_id=2, smooth=1e-5):
    """
    pred: [B, C, Z, Y, X] logits
    target: [B, Z, Y, X]
    """
    pred = torch.argmax(pred, dim=1)

    pred_bin = (pred == class_id).float()
    target_bin = (target == class_id).float()

    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum()

    return (2.0 * intersection + smooth) / (union + smooth)
