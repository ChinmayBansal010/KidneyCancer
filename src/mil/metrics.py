import torch
from sklearn.metrics import roc_auc_score

def accuracy(pred_logits, targets):
    preds = torch.argmax(pred_logits, dim=1)
    return (preds == targets).float().mean().item()

def auc_score(pred_logits, targets):
    """
    Binary AUC using probability of class 1
    """
    probs = torch.softmax(pred_logits, dim=1)[:, 1].detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    # Guard: AUC undefined if only one class present
    if len(set(targets.tolist())) < 2:
        return float("nan")

    return roc_auc_score(targets, probs)
