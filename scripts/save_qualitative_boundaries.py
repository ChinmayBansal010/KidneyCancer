"""Save qualitative boundary overlays for a trained segmentation checkpoint."""

from pathlib import Path

import torch
from tqdm import tqdm

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from kidneycancer.datasets.tumor_segmentation_dataset import TumorSegmentationDataset
from kidneycancer.metrics import dice_score, hd95, iou_score
from kidneycancer.metrics.utils import to_numpy
from kidneycancer.models.unetpp_3d import UNetPP3D
from kidneycancer.utils.visualization import save_boundary_overlay


def main() -> None:
    """Export boundary visualization images and per-case metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = Path("experiments/seg_unetpp/checkpoints/epoch_010.pth")
    checkpoint_name = checkpoint_path.stem

    model = UNetPP3D(in_channels=1, num_classes=3, base_ch=16).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dataset = TumorSegmentationDataset(
        root_dir="data/tumor_roi/kits19",
        mode="val",
        patch_size=(96, 96, 96),
    )

    output_root = Path("experiments/seg_unetpp/qualitative") / checkpoint_name
    output_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for index in tqdm(range(min(10, len(dataset)))):
            inputs, targets = dataset[index]
            inputs = inputs.unsqueeze(0).to(device)
            targets = targets.to(device)

            logits = model(inputs)
            predictions = torch.argmax(logits, dim=1)
            dice_value = dice_score(predictions, targets).item()
            iou_value = iou_score(predictions, targets).item()
            hd95_value = hd95(to_numpy(predictions[0]), to_numpy(targets))

            slice_index = inputs.shape[2] // 2
            case_dir = output_root / f"case_{index:05d}"
            case_dir.mkdir(exist_ok=True)

            save_boundary_overlay(
                to_numpy(inputs[0, 0, slice_index]),
                to_numpy(targets[slice_index]),
                to_numpy(predictions[0, slice_index]),
                dice_value,
                iou_value,
                hd95_value,
                case_dir / f"slice_{slice_index:03d}.png",
            )

            with open(case_dir / "metrics.txt", "w", encoding="utf-8") as handle:
                handle.write(
                    f"Checkpoint : {checkpoint_name}\n"
                    f"Dice       : {dice_value:.4f}\n"
                    f"IoU        : {iou_value:.4f}\n"
                    f"HD95 (vox) : {hd95_value:.2f}\n"
                )


if __name__ == "__main__":
    main()
