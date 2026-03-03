import torch
from pathlib import Path
from tqdm import tqdm

from src.models.unetpp_3d import UNetPP3D
from src.datasets.tumor_segmentation_dataset import TumorSegmentationDataset
from src.metrics import dice_score, iou_score, hd95
from src.metrics.utils import to_numpy
from src.utils.visualization import save_boundary_overlay


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    CKPT_PATH = "experiments/seg_unetpp/checkpoints/epoch_010.pth"
    CKPT_NAME = Path(CKPT_PATH).stem

    model = UNetPP3D(
        in_channels=1,
        num_classes=3,
        base_ch=16
    ).to(device)

    model.load_state_dict(
        torch.load(CKPT_PATH, map_location=device)
    )
    model.eval()

    ds = TumorSegmentationDataset(
        root_dir="data/tumor_roi/kits19",
        mode="val",
        patch_size=(96, 96, 96)
    )

    out_root = Path("experiments/seg_unetpp/qualitative") / CKPT_NAME
    out_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i in tqdm(range(min(10, len(ds)))):
            x, y = ds[i]
            x = x.unsqueeze(0).to(device)
            y = y.to(device)

            logits = model(x)
            pred = torch.argmax(logits, dim=1)

            dice = dice_score(pred, y).item()
            iou = iou_score(pred, y).item()
            hd = hd95(to_numpy(pred[0]), to_numpy(y))

            z = x.shape[2] // 2

            ct_slice = to_numpy(x[0, 0, z])
            gt_slice = to_numpy(y[z])
            pred_slice = to_numpy(pred[0, z])

            case_dir = out_root / f"case_{i:05d}"
            case_dir.mkdir(exist_ok=True)

            save_boundary_overlay(
                ct_slice,
                gt_slice,
                pred_slice,
                dice,
                iou,
                hd,
                case_dir / f"slice_{z:03d}.png"
            )

            with open(case_dir / "metrics.txt", "w") as f:
                f.write(
                    f"Checkpoint : {CKPT_NAME}\n"
                    f"Dice       : {dice:.4f}\n"
                    f"IoU        : {iou:.4f}\n"
                    f"HD95 (vox) : {hd:.2f}\n"
                )


if __name__ == "__main__":
    run()