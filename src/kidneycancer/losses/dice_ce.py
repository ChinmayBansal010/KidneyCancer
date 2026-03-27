import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def dice_loss(self, logits, targets):
        probs = F.softmax(logits, dim=1)

        targets_onehot = F.one_hot(targets, probs.shape[1])
        targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)

        intersection = torch.sum(probs * targets_onehot, dims)
        union = torch.sum(probs + targets_onehot, dims)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice_loss(logits, targets)
