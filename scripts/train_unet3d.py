"""Train a 3D U-Net baseline for kidney tumor segmentation."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from kidneycancer.datasets.tumor_segmentation_dataset import TumorSegmentationDataset
from kidneycancer.losses.dice_ce import DiceCELoss
from kidneycancer.models.unet3d import UNet3D
from kidneycancer.train.train_segmentation import train_one_epoch, validate
from kidneycancer.utils.logging_utils import configure_logging


LOGGER = configure_logging("kidneycancer.train_unet3d")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for 3D U-Net training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/tumor_roi/kits19"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--samples-per-case", type=int, default=6)
    parser.add_argument("--jitter-radius", type=int, default=12)
    return parser.parse_args(argv)


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")

    train_dataset = TumorSegmentationDataset(
        root_dir=str(args.data_root),
        mode="train",
        samples_per_case=args.samples_per_case,
        jitter_radius=args.jitter_radius,
    )
    val_dataset = TumorSegmentationDataset(root_dir=str(args.data_root), mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader


def main(argv: list[str] | None = None) -> int:
    """Train the model and save the best checkpoint."""
    args = parse_args(argv)
    try:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        LOGGER.info("Using device: %s", device)
        LOGGER.info("Training data root: %s", args.data_root)

        train_loader, val_loader = build_dataloaders(args)
        model = UNet3D().to(device)
        criterion = DiceCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        scaler = torch.amp.GradScaler(device="cuda", enabled=(device == "cuda"))

        best_val_loss = float("inf")
        for epoch in range(args.epochs):
            LOGGER.info("Epoch %s/%s started", epoch + 1, args.epochs)
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
            )
            val_loss, val_dice = validate(model, val_loader, criterion, device)

            LOGGER.info(
                "Epoch %s | Train Loss: %.4f | Val Loss: %.4f | Tumor Dice: %.4f",
                epoch + 1,
                train_loss,
                val_loss,
                val_dice,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = args.checkpoint_dir / "unet3d_best.pth"
                torch.save(model.state_dict(), checkpoint_path)
                LOGGER.info("Saved new best model to %s", checkpoint_path)
    except Exception:
        LOGGER.exception("3D U-Net training failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
