import torch
import torch.nn as nn
import torchvision.models as models
from src.models.dann import DomainHead
import timm


class MILAttention(nn.Module):
    def __init__(self, d, L=256):
        super().__init__()
        self.V = nn.Linear(d, L)
        self.U = nn.Linear(d, L)
        self.w = nn.Linear(L, 1)

    def forward(self, h):
        A = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))
        A = self.w(A)
        A = torch.softmax(A, dim=0)
        Z = torch.sum(A * h, dim=0)
        return Z, A


class MILNet(nn.Module):

    def __init__(self, num_classes=3, ssl_path=None):

        super().__init__()

        # -------- Encoder (EfficientNet-B0) --------
        self.encoder = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            in_chans=5,
            num_classes=0   # remove classifier
        )

        feat_dim = self.encoder.num_features

        # Load SSL weights if provided
        if ssl_path is not None:
            state = torch.load(ssl_path, map_location="cpu")
            self.encoder.load_state_dict(state, strict=False)
            print("Loaded SSL encoder weights.")

        # -------- MIL Attention --------
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        # -------- Classifier --------
        self.classifier = nn.Linear(feat_dim, num_classes)

        # -------- Domain head (optional) --------
        self.domain_head = DomainHead(feat_dim)

    def forward(self, slices, lambda_d=0.0):
        """
        slices: (5, 128, 128)
        """

        # Treat 5 slices as 5-channel image
        x = slices.unsqueeze(0)  # (1,5,128,128)

        feats = self.encoder(x)  # (1, feat_dim)
        feats = feats.squeeze(0)  # (feat_dim)

        cls_logits = self.classifier(feats)

        dom_logits = None
        if lambda_d > 0:
            dom_logits = self.domain_head(feats, lambda_d)

        return cls_logits, dom_logits, feats.unsqueeze(0)