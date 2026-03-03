import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


class BoundaryLoss(torch.nn.Module):
    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)[:, 1]  # tumor channel
        target = target.cpu().numpy()

        loss = 0.0
        for i in range(target.shape[0]):
            dist = distance_transform_edt(1 - target[i])
            dist = torch.tensor(dist, device=probs.device, dtype=torch.float32)
            loss += torch.mean(probs[i] * dist)

        return loss / target.shape[0]