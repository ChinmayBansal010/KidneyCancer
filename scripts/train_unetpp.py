"""Train a 3D UNet++ segmentation model."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from kidneycancer.datasets.tumor_segmentation_dataset import TumorSegmentationDataset
from kidneycancer.losses.segmentation_loss import SegmentationLoss
from kidneycancer.metrics import asd, dice_score, hd95, iou_score
from kidneycancer.metrics.utils import to_numpy
from kidneycancer.models.unetpp_3d import UNetPP3D
from kidneycancer.utils.logger import Logger
from kidneycancer.utils.logging_utils import configure_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for UNet++ training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/tumor_roi/kits19"))
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments/seg_unetpp"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--samples-per-case", type=int, default=4)
    parser.add_argument("--jitter-radius", type=int, default=12)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(96, 96, 96))
    return parser.parse_args(argv)


def aggregate_stats(stats: list[dict[str, float]]) -> dict[str, float]:
    """Average per-batch metric dictionaries into one summary."""
    return {key: float(np.mean([entry[key] for entry in stats])) for key in stats[0]}


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: SegmentationLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
) -> dict[str, float]:
    """Run one training epoch and collect aggregated loss values."""
    model.train()
    stats: list[dict[str, float]] = []

    for inputs, targets in tqdm(loader, leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(inputs)
            loss, logs = loss_fn(logits, targets, compute_boundary=True)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        stats.append(logs)

    return aggregate_stats(stats)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: SegmentationLoss,
    device: str,
) -> dict[str, float]:
    """Evaluate the model on the validation split."""
    model.eval()
    stats: list[dict[str, float]] = []
    dice_values: list[float] = []
    iou_values: list[float] = []
    hd95_values: list[float] = []
    asd_values: list[float] = []

    for inputs, targets in tqdm(loader, leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        _, logs = loss_fn(logits, targets, compute_boundary=False)
        predictions = torch.argmax(logits, dim=1)

        dice_values.append(dice_score(predictions, targets).item())
        iou_values.append(iou_score(predictions, targets).item())

        pred_np = to_numpy(predictions[0])
        target_np = to_numpy(targets[0])
        pred_tumor = pred_np == 2
        target_tumor = target_np == 2

        if pred_tumor.sum() == 0 or target_tumor.sum() == 0:
            hd95_values.append(np.nan)
            asd_values.append(np.nan)
        else:
            hd95_values.append(hd95(pred_tumor, target_tumor))
            asd_values.append(asd(pred_tumor, target_tumor))

        stats.append(logs)

    aggregated = aggregate_stats(stats)
    aggregated["dice"] = float(np.mean(dice_values))
    aggregated["iou"] = float(np.mean(iou_values))
    aggregated["hd95"] = float(np.nanmean(hd95_values))
    aggregated["asd"] = float(np.nanmean(asd_values))
    return aggregated


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders for UNet++ experiments."""
    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")

    patch_size = tuple(args.patch_size)
    train_dataset = TumorSegmentationDataset(
        root_dir=str(args.data_root),
        mode="train",
        patch_size=patch_size,
        samples_per_case=args.samples_per_case,
        jitter_radius=args.jitter_radius,
    )
    val_dataset = TumorSegmentationDataset(
        root_dir=str(args.data_root),
        mode="val",
        patch_size=patch_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    return train_loader, val_loader


def main(argv: list[str] | None = None) -> int:
    """Train the UNet++ model and save checkpoints and metrics."""
    args = parse_args(argv)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = True

    checkpoint_dir = args.experiment_root / "checkpoints"
    log_dir = args.experiment_root / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_logger = configure_logging(
        "kidneycancer.train_unetpp",
        log_file=log_dir / "train.log",
    )
    csv_logger = Logger(log_dir / "train_log.csv")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        run_logger.info("Using device: %s", device)
        run_logger.info("Training data root: %s", args.data_root)
        run_logger.info("Experiment root: %s", args.experiment_root)

        train_loader, val_loader = build_dataloaders(args)
        model = UNetPP3D(in_channels=1, num_classes=3, base_ch=16).to(device)
        loss_fn = SegmentationLoss(num_classes=3, use_boundary=True, w_boundary=0.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

        best_dice = 0.0
        for epoch in range(1, args.epochs + 1):
            run_logger.info("Epoch %s/%s started", epoch, args.epochs)
            train_stats = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                scaler,
                device,
            )
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
            csv_logger.log(row)
            run_logger.info("Metrics: %s", row)

            if val_stats["dice"] > best_dice:
                best_dice = val_stats["dice"]
                checkpoint_path = checkpoint_dir / "best_model.pth"
                torch.save(model.state_dict(), checkpoint_path)
                run_logger.info("Saved best model to %s with Dice %.4f", checkpoint_path, best_dice)

            if epoch % 10 == 0:
                checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
                torch.save(model.state_dict(), checkpoint_path)
                run_logger.info("Saved periodic checkpoint to %s", checkpoint_path)
    except Exception:
        run_logger.exception("UNet++ training failed")
        csv_logger.close()
        return 1

    csv_logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
