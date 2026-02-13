import torch
from torch.utils.data import DataLoader
from src.metrics.dice import dice_score
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0

    for x, y in tqdm(loader, desc="Train", dynamic_ncols=True):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=(device == "cuda")):
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)



@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    count = 0

    for x, y in tqdm(loader, desc="Val", dynamic_ncols=True):
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        loss = criterion(out, y)

        dice = dice_score(out, y, class_id=2)

        total_loss += loss.item()
        total_dice += dice.item()
        count += 1

    return total_loss / count, total_dice / count