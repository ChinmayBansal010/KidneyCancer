import torch
from tqdm import tqdm

from .metrics import accuracy, auc_score


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    all_logits = []
    all_targets = []

    for bags, labels, _ in tqdm(loader, desc="Train", leave=False):
        bags = bags.to(device)        # [B, N, 1, H, W]
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits, attn = model(bags)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_logits.append(logits.detach())
        all_targets.append(labels.detach())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    return {
        "loss": total_loss / len(loader),
        "acc": accuracy(all_logits, all_targets),
        "auc": auc_score(all_logits, all_targets),
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_logits = []
    all_targets = []
    attention_log = []

    for bags, labels, case_ids in tqdm(loader, desc="Val", leave=False):
        bags = bags.to(device)
        labels = labels.to(device)

        logits, attn = model(bags)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        all_logits.append(logits)
        all_targets.append(labels)

        # Save attention weights for interpretability
        attention_log.append({
            "case_id": case_ids[0],
            "attention": attn.squeeze().cpu().numpy().tolist()
        })

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    return {
        "loss": total_loss / len(loader),
        "acc": accuracy(all_logits, all_targets),
        "auc": auc_score(all_logits, all_targets),
        "attention": attention_log
    }
