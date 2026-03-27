import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [B, C, Z, Y, X]
        targets: [B, Z, Y, X] with class indices {0..C-1}
        """

        probs = F.softmax(logits, dim=1)

        # one-hot encode targets
        targets_oh = F.one_hot(
            targets, num_classes=self.num_classes
        )  # [B,Z,Y,X,C]
        targets_oh = targets_oh.permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)

        intersection = torch.sum(probs * targets_oh, dims)
        cardinality = torch.sum(probs + targets_oh, dims)

        dice = (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth
        )

        # exclude background (class 0)
        dice = dice[1:]

        return 1.0 - dice.mean()