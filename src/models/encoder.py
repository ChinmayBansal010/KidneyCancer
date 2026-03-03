# src/models/encoder.py
import timm
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = timm.create_model(
            "efficientnet_b7",
            pretrained=True,
            in_chans=5,
            num_classes=0
        )

    def forward(self, x):
        return self.net(x)