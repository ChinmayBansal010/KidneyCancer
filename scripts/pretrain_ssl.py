# scripts/pretrain_ssl.py

import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from torch.amp import autocast, GradScaler

from src.datasets.tcga_mil_dataset import TCGAMILDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
BATCH = 8
LR = 1e-4
MASK_RATIO = 0.3

SAVE_PATH = Path("experiments/ssl_b0_encoder.pth")
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)


# ----------------------------
# SSL Model
# ----------------------------
class SSLModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = timm.create_model(
            "efficientnet_b0",
            in_chans=5,
            pretrained=False,
            num_classes=0
        )

        self.decoder = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 5 * 128 * 128)
        )

    def forward(self, x):
        feats = self.encoder(x)
        recon = self.decoder(feats)
        recon = recon.view(-1, 5, 128, 128)
        return recon


def random_mask(x):
    B, C, H, W = x.shape
    mask = torch.rand_like(x) < MASK_RATIO
    x = x.clone()
    x[mask] = 0
    return x


# ----------------------------
# DATA
# ----------------------------
train_ds = TCGAMILDataset(split="train")
val_ds = TCGAMILDataset(split="val")

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH)

print("Train size:", len(train_ds))
print("Val size:", len(val_ds))

# ----------------------------
# TRAIN
# ----------------------------
model = SSLModel().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.MSELoss()
scaler = GradScaler()

best_val = 1e9

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for x, _ in tqdm(train_loader, desc=f"Train {epoch+1}"):

        x = x.to(DEVICE)

        masked = random_mask(x)

        optimizer.zero_grad()

        with autocast("cuda"):
            recon = model(masked)
            loss = criterion(recon, x)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---- Validation ----
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(DEVICE)

            masked = random_mask(x)

            with autocast("cuda"):
                recon = model(masked)
                loss = criterion(recon, x)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"\nEpoch {epoch+1}")
    print("Train Loss:", round(train_loss, 4))
    print("Val Loss:", round(val_loss, 4))

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.encoder.state_dict(), SAVE_PATH)
        print("Saved best encoder")

print("SSL training complete.")