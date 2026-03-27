"""Train the MIL classifier on TCGA 2.5D slices."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from kidneycancer.utils.logging_utils import configure_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for MIL training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/tcga_2p5d"))
    parser.add_argument("--ssl-weights", type=Path, default=Path("experiments/ssl_b0_encoder.pth"))
    parser.add_argument("--save-path", type=Path, default=Path("experiments/mil_b0_best.pth"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def set_seed(seed: int = 42) -> None:
    """Make training runs reproducible enough for experiment tracking."""
    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(argv: list[str] | None = None) -> int:
    """Train the MIL model and save the best checkpoint by macro AUROC."""
    args = parse_args(argv)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(
        "kidneycancer.train_mil",
        log_file=args.save_path.parent / "train_mil.log",
    )

    try:
        import numpy as np
        import torch
        import torch.optim as optim
        from sklearn.metrics import accuracy_score, roc_auc_score
        from torch.amp import GradScaler, autocast
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        from kidneycancer.datasets.tcga_mil_dataset import TCGAMILDataset
        from kidneycancer.losses.focal import focal
        from kidneycancer.models.mil_model import MILNet

        def build_dataloaders(data_root: Path) -> tuple[DataLoader, DataLoader]:
            if not data_root.exists():
                raise FileNotFoundError(f"Data root does not exist: {data_root}")

            train_dataset = TCGAMILDataset(str(data_root), split="train")
            val_dataset = TCGAMILDataset(str(data_root), split="val")
            logger.info("Train size: %s", len(train_dataset))
            logger.info("Val size: %s", len(val_dataset))
            return (
                DataLoader(train_dataset, batch_size=1, shuffle=True),
                DataLoader(val_dataset, batch_size=1, shuffle=False),
            )

        def compute_class_weights(device: str) -> torch.Tensor:
            class_counts = torch.tensor([51, 6, 7], dtype=torch.float32)
            weights = 1.0 / class_counts
            weights = weights / weights.sum()
            return weights.to(device)

        def train_epoch(
            model: torch.nn.Module,
            loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            scaler: GradScaler,
            weights: torch.Tensor,
            device: str,
        ) -> float:
            model.train()
            total_loss = 0.0

            for slices, label in tqdm(loader, desc="Train", leave=False):
                slices = slices.squeeze(0).to(device)
                label = label.to(device)

                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=(device == "cuda")):
                    logits, _, _ = model(slices)
                    loss = focal(logits.unsqueeze(0), label, weights=weights)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

            return total_loss / len(loader)

        @torch.no_grad()
        def validate(
            model: torch.nn.Module,
            loader: DataLoader,
            weights: torch.Tensor,
            device: str,
        ) -> dict[str, float]:
            model.eval()
            all_probabilities: list[np.ndarray] = []
            all_labels: list[int] = []
            total_loss = 0.0

            for slices, label in loader:
                slices = slices.squeeze(0).to(device)
                label = label.to(device)

                logits, _, _ = model(slices)
                probabilities = torch.softmax(logits, dim=0).cpu().numpy()
                all_probabilities.append(probabilities)
                all_labels.append(label.item())
                total_loss += focal(logits.unsqueeze(0), label, weights=weights).item()

            probability_array = np.array(all_probabilities)
            labels_array = np.array(all_labels)
            y_true = np.eye(3)[labels_array]

            aucs = []
            for class_index in range(3):
                try:
                    aucs.append(
                        roc_auc_score(y_true[:, class_index], probability_array[:, class_index])
                    )
                except ValueError:
                    aucs.append(0.5)

            predictions = np.argmax(probability_array, axis=1)
            return {
                "val_loss": total_loss / len(loader),
                "accuracy": accuracy_score(labels_array, predictions),
                "macro_auc": float(np.mean(aucs)),
                "auc_kirc": float(aucs[0]),
                "auc_kirp": float(aucs[1]),
                "auc_kich": float(aucs[2]),
            }

        device = "cuda" if torch.cuda.is_available() else "cpu"
        set_seed(args.seed)
        logger.info("Using device: %s", device)
        logger.info("Data root: %s", args.data_root)

        train_loader, val_loader = build_dataloaders(args.data_root)
        weights = compute_class_weights(device)
        ssl_weights = str(args.ssl_weights) if args.ssl_weights.exists() else None
        if ssl_weights is None:
            logger.warning(
                "SSL weights not found at %s; training without pretrained encoder",
                args.ssl_weights,
            )

        model = MILNet(num_classes=3, ssl_path=ssl_weights).to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = GradScaler(enabled=(device == "cuda"))

        best_auc = 0.0
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, scaler, weights, device)
            metrics = validate(model, val_loader, weights, device)
            scheduler.step()

            logger.info(
                "Epoch %s | train_loss=%.4f val_loss=%.4f acc=%.4f "
                "auc_kirc=%.4f auc_kirp=%.4f auc_kich=%.4f macro_auc=%.4f",
                epoch + 1,
                train_loss,
                metrics["val_loss"],
                metrics["accuracy"],
                metrics["auc_kirc"],
                metrics["auc_kirp"],
                metrics["auc_kich"],
                metrics["macro_auc"],
            )

            if metrics["macro_auc"] > best_auc:
                best_auc = metrics["macro_auc"]
                torch.save(model.state_dict(), args.save_path)
                logger.info("Saved best model to %s", args.save_path)
    except Exception:
        logger.exception("MIL training failed")
        return 1

    logger.info("Training complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
