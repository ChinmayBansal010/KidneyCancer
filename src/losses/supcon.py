# src/losses/supcon.py
import torch
import torch.nn.functional as F

def supcon(z, y, T=0.07):
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.T) / T
    mask = y.unsqueeze(1) == y.unsqueeze(0)
    exp = torch.exp(sim)
    return -torch.log(
        exp[mask].sum(1) / exp.sum(1)
    ).mean()