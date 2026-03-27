# src/models/dann.py
import torch.nn as nn

from .grl import GRL

class DomainHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d,256),
            nn.ReLU(),
            nn.Linear(256,2)
        )

    def forward(self, x, l=1.0):
        x = GRL.apply(x, l)
        return self.fc(x)
