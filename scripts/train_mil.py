# scripts/train_mil.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import random

from src.models.mil_model import MILNet
from src.datasets.tcga_mil_dataset import TCGAMILDataset
from src.losses.focal import focal
from src.losses.supcon import supcon


# -----------------------------------
# CONFIG
# -----------------------------------
DATA_ROOT = "data/tcga_2p5d"
SSL_WEIGHTS = "experiments/ssl_b0_encoder.pth"
SAVE_PATH = "experiments/mil_b0_best.pth"

EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-4

LAMBDA_SUP = 0   # reduce, 0.5 was too aggressive

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# -----------------------------------
# DATA
# -----------------------------------
train_ds = TCGAMILDataset(DATA_ROOT, split="train")
val_ds = TCGAMILDataset(DATA_ROOT, split="val")

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

class_counts = torch.tensor([51, 6, 7], dtype=torch.float32)
weights = 1.0 / class_counts
weights = weights / weights.sum()
weights = weights.to(device)

print("Train size:", len(train_ds))
print("Val size:", len(val_ds))


# -----------------------------------
# MODEL
# -----------------------------------
model = MILNet(num_classes=3, ssl_path=SSL_WEIGHTS).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

scaler = GradScaler()

best_auc = 0


# -----------------------------------
# TRAIN LOOP
# -----------------------------------
for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for slices, label in tqdm(train_loader, desc=f"Train {epoch+1}"):

        # slices: (1, K, 128,128)
        slices = slices.squeeze(0).to(device)  # (K,128,128)
        label = label.to(device)

        optimizer.zero_grad()

        with autocast("cuda"):

            cls_logits, _, feats = model(slices)
            # cls_logits: (3,)
            # feats: (K, D)

            L_cls = focal(cls_logits.unsqueeze(0), label, weights=weights)

            loss = L_cls 

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    scheduler.step()

    # -----------------------------------
    # VALIDATION
    # -----------------------------------
    model.eval()
    all_probs = []
    all_labels = []
    val_loss = 0

    with torch.no_grad():
        for slices, label in val_loader:

            slices = slices.squeeze(0).to(device)
            label = label.to(device)

            cls_logits, _, feats = model(slices)

            probs = torch.softmax(cls_logits, dim=0).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(label.item())

            val_loss += focal(cls_logits.unsqueeze(0), label, weights=weights).item()

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # One-hot encoding
    y_true = np.eye(3)[all_labels]

    aucs = []
    for c in range(3):
        try:
            auc = roc_auc_score(y_true[:, c], all_probs[:, c])
        except:
            auc = 0.5
        aucs.append(auc)

    macro_auc = np.mean(aucs)

    preds = np.argmax(all_probs, axis=1)
    acc = accuracy_score(all_labels, preds)

    print("\n-----------------------------------")
    print(f"Epoch {epoch+1}")
    print(f"Train Loss : {train_loss/len(train_loader):.4f}")
    print(f"Val Loss   : {val_loss/len(val_loader):.4f}")
    print(f"Accuracy   : {acc:.4f}")
    print(f"AUROC KIRC : {aucs[0]:.4f}")
    print(f"AUROC KIRP : {aucs[1]:.4f}")
    print(f"AUROC KICH : {aucs[2]:.4f}")
    print(f"Macro AUROC: {macro_auc:.4f}")
    print("-----------------------------------\n")

    if macro_auc > best_auc:
        best_auc = macro_auc
        torch.save(model.state_dict(), SAVE_PATH)
        print("Saved best model by AUROC.")

print("Training complete.")