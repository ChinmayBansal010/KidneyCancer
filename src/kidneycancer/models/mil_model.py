"""MIL model definitions for TCGA classification experiments."""

import logging

import timm
import torch
import torch.nn as nn

from .dann import DomainHead


LOGGER = logging.getLogger(__name__)


class MILAttention(nn.Module):
    """Attention block for aggregating instance-level MIL features."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.value = nn.Linear(feature_dim, hidden_dim)
        self.gate = nn.Linear(feature_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention = torch.tanh(self.value(features)) * torch.sigmoid(self.gate(features))
        attention = self.score(attention)
        attention = torch.softmax(attention, dim=0)
        pooled = torch.sum(attention * features, dim=0)
        return pooled, attention


class MILNet(nn.Module):
    """MIL classifier built on top of an EfficientNet encoder."""

    def __init__(self, num_classes: int = 3, ssl_path: str | None = None):
        super().__init__()
        self.encoder = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            in_chans=5,
            num_classes=0,
        )
        feature_dim = self.encoder.num_features

        if ssl_path is not None:
            state_dict = torch.load(ssl_path, map_location="cpu")
            self.encoder.load_state_dict(state_dict, strict=False)
            LOGGER.info("Loaded SSL encoder weights from %s", ssl_path)

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.domain_head = DomainHead(feature_dim)

    def forward(
        self,
        slices: torch.Tensor,
        lambda_d: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Predict class logits and optional domain logits for a 5-slice input."""
        inputs = slices.unsqueeze(0)
        features = self.encoder(inputs).squeeze(0)
        class_logits = self.classifier(features)

        domain_logits = None
        if lambda_d > 0:
            domain_logits = self.domain_head(features, lambda_d)

        return class_logits, domain_logits, features.unsqueeze(0)
