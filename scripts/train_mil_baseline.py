import os
import json
import random
import logging
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.mil.dataset import MILSliceDataset
from src.mil.model import AttentionMIL
from src.mil.train import train_one_epoch, validate

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MIL")


# ---------------- Stratified Case Split ----------------
def stratified_case_split(dataset, val_ratio=0.2, seed=42):
    """
    Split cases by label to guarantee both classes in val set.
    """
    random.seed(seed)

    label_to_cases = defaultdict(list)
    for case in dataset.cases:
        with open(case / "slice_meta.json") as f:
            meta = json.load(f)
        label = int(meta["tumor_present"])
        label_to_cases[label].append(case)

    train_cases, val_cases = [], []

    for label, cases in label_to_cases.items():
        random.shuffle(cases)
        n_val = max(1, int(len(cases) * val_ratio))
        val_cases.extend(cases[:n_val])
        train_cases.extend(cases[n_val:])

    random.shuffle(train_cases)
    random.shuffle(val_cases)

    return train_cases, val_cases


# ---------------- Main ----------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load full dataset once
    full_ds = MILSliceDataset(
        root_dir="data/slices_2p5d/kits19",
        label_source="kits19"
    )

    # Stratified split (MANDATORY for MIL)
    train_cases, val_cases = stratified_case_split(full_ds, val_ratio=0.2)

    logger.info(f"Train cases: {len(train_cases)} | Val cases: {len(val_cases)}")

    # Build train dataset
    train_ds = MILSliceDataset(
        root_dir="data/slices_2p5d/kits19",
        label_source="kits19"
    )
    train_ds.cases = train_cases

    # Build val dataset
    val_ds = MILSliceDataset(
        root_dir="data/slices_2p5d/kits19",
        label_source="kits19"
    )
    val_ds.cases = val_cases

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # Model
    model = AttentionMIL(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    EPOCHS = 30
    best_auc = -1.0

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # ---------------- Training Loop ----------------
    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_metrics = validate(
            model, val_loader, criterion, device
        )

        logger.info(
            f"Train | Loss {train_metrics['loss']:.4f} "
            f"Acc {train_metrics['acc']:.3f} "
            f"AUC {train_metrics['auc']:.3f}"
        )

        logger.info(
            f"Val   | Loss {val_metrics['loss']:.4f} "
            f"Acc {val_metrics['acc']:.3f} "
            f"AUC {val_metrics['auc']:.3f}"
        )

        # Save best model by AUC (not accuracy)
        if not torch.isnan(torch.tensor(val_metrics["auc"])) and val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]

            torch.save(
                model.state_dict(),
                "checkpoints/mil_best.pth"
            )

            with open("logs/attention_best.json", "w") as f:
                json.dump(val_metrics["attention"], f, indent=2)

            logger.info("Saved new best MIL model + attention weights")

    logger.info("MIL training complete")


if __name__ == "__main__":
    main()
