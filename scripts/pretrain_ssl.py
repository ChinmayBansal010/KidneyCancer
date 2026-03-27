"""Pretrain an EfficientNet encoder with a simple reconstruction objective."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from kidneycancer.utils.logging_utils import configure_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for SSL pretraining."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/tcga_2p5d"))
    parser.add_argument("--save-path", type=Path, default=Path("experiments/ssl_b0_encoder.pth"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Train the SSL model and save the best encoder checkpoint."""
    args = parse_args(argv)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(
        "kidneycancer.pretrain_ssl",
        log_file=args.save_path.parent / "pretrain_ssl.log",
    )

    try:
        import timm
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.amp import GradScaler, autocast
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        from kidneycancer.datasets.tcga_mil_dataset import TCGAMILDataset

        class SSLModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = timm.create_model(
                    "efficientnet_b0",
                    in_chans=5,
                    pretrained=False,
                    num_classes=0,
                )
                self.decoder = nn.Sequential(
                    nn.Linear(1280, 512),
                    nn.ReLU(),
                    nn.Linear(512, 5 * 128 * 128),
                )

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                features = self.encoder(inputs)
                reconstruction = self.decoder(features)
                return reconstruction.view(-1, 5, 128, 128)

        def random_mask(inputs: torch.Tensor, mask_ratio: float) -> torch.Tensor:
            mask = torch.rand_like(inputs) < mask_ratio
            masked_inputs = inputs.clone()
            masked_inputs[mask] = 0
            return masked_inputs

        def build_dataloaders(data_root: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
            if not data_root.exists():
                raise FileNotFoundError(f"Data root does not exist: {data_root}")

            train_dataset = TCGAMILDataset(root=str(data_root), split="train")
            val_dataset = TCGAMILDataset(root=str(data_root), split="val")
            logger.info("Train size: %s", len(train_dataset))
            logger.info("Val size: %s", len(val_dataset))
            return (
                DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                DataLoader(val_dataset, batch_size=batch_size),
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", device)
        logger.info("Data root: %s", args.data_root)

        train_loader, val_loader = build_dataloaders(args.data_root, args.batch_size)
        model = SSLModel().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        criterion = nn.MSELoss()
        scaler = GradScaler(enabled=(device == "cuda"))

        best_val_loss = float("inf")
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0

            for inputs, _ in tqdm(train_loader, desc=f"Train {epoch + 1}", leave=False):
                inputs = inputs.to(device)
                masked_inputs = random_mask(inputs, args.mask_ratio)
                optimizer.zero_grad(set_to_none=True)

                with autocast("cuda", enabled=(device == "cuda")):
                    reconstruction = model(masked_inputs)
                    loss = criterion(reconstruction, inputs)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, _ in val_loader:
                    inputs = inputs.to(device)
                    masked_inputs = random_mask(inputs, args.mask_ratio)
                    with autocast("cuda", enabled=(device == "cuda")):
                        reconstruction = model(masked_inputs)
                        loss = criterion(reconstruction, inputs)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            logger.info(
                "Epoch %s | train_loss=%.4f val_loss=%.4f",
                epoch + 1,
                train_loss,
                val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.encoder.state_dict(), args.save_path)
                logger.info("Saved best encoder to %s", args.save_path)
    except Exception:
        logger.exception("SSL pretraining failed")
        return 1

    logger.info("SSL training complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
