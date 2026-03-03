import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.unetpp_3d import UNetPP3D
from src.losses.segmentation_loss import SegmentationLoss
from src.utils.logger import Logger
from src.datasets.tumor_segmentation_dataset import TumorSegmentationDataset
from src.metrics import dice_score, iou_score, hd95, asd
from src.metrics.utils import to_numpy


def train_one_epoch(model, loader, loss_fn, opt, scaler, device):
    model.train()
    stats = []

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)

        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(x)
            loss, logs = loss_fn(logits, y, compute_boundary=True)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        stats.append(logs)

    return aggregate_stats(stats)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    import gc

    model.eval()
    stats = []
    dice_list, iou_list, hd_list, asd_list = [], [], [], []

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss, logs = loss_fn(logits, y, compute_boundary=False)

        pred = torch.argmax(logits, dim=1)

        dice_list.append(dice_score(pred, y).item())
        iou_list.append(iou_score(pred, y).item())

        pred_np = to_numpy(pred[0])
        gt_np = to_numpy(y[0])

        pred_tumor = pred_np == 2
        gt_tumor = gt_np == 2

        if pred_tumor.sum() == 0 or gt_tumor.sum() == 0:
            hd_list.append(np.nan)
            asd_list.append(np.nan)
        else:
            hd_list.append(hd95(pred_tumor, gt_tumor))
            asd_list.append(asd(pred_tumor, gt_tumor))

        stats.append(logs)

        del pred, logits, x, y, pred_np, gt_np
        gc.collect()

    agg = aggregate_stats(stats)
    agg["dice"] = float(np.mean(dice_list))
    agg["iou"] = float(np.mean(iou_list))
    agg["hd95"] = float(np.nanmean(hd_list))
    agg["asd"] = float(np.nanmean(asd_list))
    return agg


def aggregate_stats(stats):
    out = {}
    for k in stats[0]:
        out[k] = float(np.mean([s[k] for s in stats]))
    return out


# -------------------------
# Main
# -------------------------
def train():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------------------------
    # Directories
    # -------------------------
    exp_root = Path("experiments/seg_unetpp")
    (exp_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_root / "logs").mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Datasets
    # -------------------------
    train_ds = TumorSegmentationDataset(
        root_dir="data/tumor_roi/kits19",
        mode="train",
        patch_size=(96, 96, 96),
        samples_per_case=4,
        jitter_radius=12,
    )

    val_ds = TumorSegmentationDataset(
        root_dir="data/tumor_roi/kits19", mode="val", patch_size=(96, 96, 96)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # -------------------------
    # Model / Loss / Optim
    # -------------------------
    model = UNetPP3D(in_channels=1, num_classes=3, base_ch=16).to(device)

    loss_fn = SegmentationLoss(num_classes=3, use_boundary=True, w_boundary=0.5)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    log_file = exp_root / "logs" / "train_log.csv"
    logger = Logger(log_file)

    best_dice = 0.0

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(1, 101):
        print(f"\nEpoch {epoch}/100")

        train_stats = train_one_epoch(model, train_loader, loss_fn, opt, scaler, device)

        val_stats = validate(model, val_loader, loss_fn, device)

        row = {
            "epoch": epoch,
            "train_loss": train_stats["total"],
            "val_loss": val_stats["total"],
            "train_dice_loss": train_stats["dice"],
            "train_boundary_loss": train_stats["boundary"],
            "val_dice": val_stats["dice"],
            "val_iou": val_stats["iou"],
            "val_hd95": val_stats["hd95"],
            "val_asd": val_stats["asd"],
        }

        logger.log(row)
        print("Logged:", row)

        # Save best model
        if val_stats["dice"] > best_dice:
            best_dice = val_stats["dice"]
            torch.save(model.state_dict(), exp_root / "checkpoints/best_model.pth")
            print(f"✓ Saved best model (Dice={best_dice:.4f})")

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save(
                model.state_dict(), exp_root / f"checkpoints/epoch_{epoch:03d}.pth"
            )

    logger.close()


if __name__ == "__main__":
    train()
