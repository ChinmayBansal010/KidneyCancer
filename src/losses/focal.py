import torch
import torch.nn.functional as F

def focal(logits, y, weights=None, g=2.0, eps=1e-6):

    p = F.softmax(logits, dim=1)
    pt = p[range(len(y)), y]
    pt = torch.clamp(pt, eps, 1.0)

    if weights is not None:
        w = weights[y]
    else:
        w = 1.0

    return -(w * (1 - pt) ** g * torch.log(pt)).mean()