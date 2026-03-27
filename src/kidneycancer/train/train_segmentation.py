"""Shared training utilities for segmentation experiments."""

import torch
from tqdm import tqdm

from kidneycancer.metrics.dice import dice_score


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: str,
    scaler: torch.amp.GradScaler,
) -> float:
    """Train a segmentation model for one epoch."""
    model.train()
    total_loss = 0.0

    for inputs, targets in tqdm(loader, desc="Train", dynamic_ncols=True):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model: torch.nn.Module, loader, criterion, device: str) -> tuple[float, float]:
    """Evaluate a segmentation model and return loss and tumor Dice."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    count = 0

    for inputs, targets in tqdm(loader, desc="Val", dynamic_ncols=True):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        dice = dice_score(outputs, targets, class_id=2)

        total_loss += loss.item()
        total_dice += dice.item()
        count += 1

    return total_loss / count, total_dice / count
