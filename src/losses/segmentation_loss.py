from src.losses.dice_loss import DiceLoss
import torch.nn as nn


class SegmentationLoss(nn.Module):
    def __init__(self, num_classes=3, use_boundary=True, w_boundary=0.5):
        super().__init__()

        self.dice = DiceLoss(num_classes=num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.use_boundary = use_boundary
        self.w_boundary = w_boundary

        if use_boundary:
            from src.losses.boundary_loss import BoundaryLoss
            self.boundary = BoundaryLoss()

    def forward(self, logits, target, compute_boundary=True):
        loss_dice = self.dice(logits, target)
        loss_ce   = self.ce(logits, target)

        total = loss_dice + loss_ce
        logs = {
            "dice": loss_dice.item(),
            "ce": loss_ce.item(),
            "total": total.item(),
            "boundary": 0.0
        }

        if self.use_boundary and compute_boundary:
            lb = self.boundary(logits, target)
            total = total + self.w_boundary * lb
            logs["boundary"] = lb.item()
            logs["total"] = total.item()

        return total, logs