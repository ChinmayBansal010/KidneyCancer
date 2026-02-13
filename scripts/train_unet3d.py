import torch
import logging
from torch.utils.data import DataLoader

from src.datasets.tumor_segmentation_dataset import TumorSegmentationDataset
from src.models.unet3d import UNet3D
from src.losses.dice_ce import DiceCELoss
from src.train.train_segmentation import train_one_epoch, validate

# ------------------------
# Logging setup
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("train")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    scaler = torch.amp.GradScaler(device='cuda',enabled=(device == "cuda"))

    train_ds = TumorSegmentationDataset(
        root_dir="data/tumor_roi/kits19",
        mode="train",
        samples_per_case=6,
        jitter_radius=12
    )

    val_ds = TumorSegmentationDataset(
        root_dir="data/tumor_roi/kits19",
        mode="val"
    )

    # IMPORTANT: num_workers=0 for Windows stability
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = UNet3D().to(device)
    criterion = DiceCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    EPOCHS = 30
    best_val = float("inf")

    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch+1}/{EPOCHS} started")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_dice = validate(
            model, val_loader, criterion, device
        )

        logger.info(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Tumor Dice: {val_dice:.4f}"
        )

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                model.state_dict(),
                "checkpoints/unet3d_best.pth"
            )
            logger.info("Saved new best model")

if __name__ == "__main__":
    import os
    os.makedirs("checkpoints", exist_ok=True)
    main()
