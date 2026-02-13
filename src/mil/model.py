import torch
import torch.nn as nn
import torchvision.models as models


class AttentionMIL(nn.Module):
    def __init__(self, num_classes=2, feat_dim=512):
        super().__init__()

        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.attention = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, bag):
        """
        bag: [B, N, 1, H, W]
        """
        B, N, C, H, W = bag.shape

        bag = bag.view(B * N, C, H, W)
        feats = self.backbone(bag)        # [B*N, F]
        feats = feats.view(B, N, -1)      # [B, N, F]

        attn_scores = self.attention(feats)       # [B, N, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)

        bag_feat = (attn_weights * feats).sum(dim=1)
        out = self.classifier(bag_feat)

        return out, attn_weights
