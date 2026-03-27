import torch.nn as nn

from kidneycancer.losses.grl import GRL

class DomainHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.Linear(256, 2)   # 2 domains (KiTS vs TCGA)
        )

    def forward(self, x, lambda_):
        x = GRL.apply(x, lambda_)
        return self.fc(x)
