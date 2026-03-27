"""Run sliding-window inference for the UNet++ segmentation model."""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from kidneycancer.datasets.tumor_segmentation_dataset import TumorSegmentationDataset
from kidneycancer.models.unetpp_3d import UNetPP3D
from kidneycancer.utils.logging_utils import configure_logging
from kidneycancer.utils.visualization import save_overlay


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for UNet++ inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/tumor_roi/kits19"))
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/seg_unetpp/checkpoints/best_model.pth"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/seg_unetpp/inference_sw"))
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--stride", type=int, nargs=3, default=(48, 48, 48))
    return parser.parse_args(argv)


def sliding_window_inference(
    volume: torch.Tensor,
    model: torch.nn.Module,
    patch_size: tuple[int, int, int] = (96, 96, 96),
    stride: tuple[int, int, int] = (48, 48, 48),
    num_classes: int = 3,
    device: str = "cuda",
) -> torch.Tensor:
    """Predict a full volume by averaging overlapping sliding-window patches."""
    model.eval()

    _, depth, height, width = volume.shape
    patch_depth, patch_height, patch_width = patch_size
    stride_depth, stride_height, stride_width = stride

    score_map = torch.zeros((num_classes, depth, height, width), device=device)
    count_map = torch.zeros((depth, height, width), device=device)

    z_starts = list(range(0, max(depth - patch_depth, 0) + 1, stride_depth))
    y_starts = list(range(0, max(height - patch_height, 0) + 1, stride_height))
    x_starts = list(range(0, max(width - patch_width, 0) + 1, stride_width))

    if z_starts[-1] != depth - patch_depth:
        z_starts.append(depth - patch_depth)
    if y_starts[-1] != height - patch_height:
        y_starts.append(height - patch_height)
    if x_starts[-1] != width - patch_width:
        x_starts.append(width - patch_width)

    with torch.no_grad():
        for z_index in z_starts:
            for y_index in y_starts:
                for x_index in x_starts:
                    patch = volume[
                        :,
                        z_index : z_index + patch_depth,
                        y_index : y_index + patch_height,
                        x_index : x_index + patch_width,
                    ].unsqueeze(0).to(device)
                    logits = model(patch)
                    probabilities = torch.softmax(logits, dim=1)[0]
                    score_map[
                        :,
                        z_index : z_index + patch_depth,
                        y_index : y_index + patch_height,
                        x_index : x_index + patch_width,
                    ] += probabilities
                    count_map[
                        z_index : z_index + patch_depth,
                        y_index : y_index + patch_height,
                        x_index : x_index + patch_width,
                    ] += 1

    score_map /= torch.clamp(count_map, min=1.0).unsqueeze(0)
    return torch.argmax(score_map, dim=0).cpu()


def main(argv: list[str] | None = None) -> int:
    """Generate qualitative inference outputs for a validation subset."""
    args = parse_args(argv)
    logger = configure_logging("kidneycancer.infer_unetpp")

    try:
        if not args.data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {args.data_root}")
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", device)
        args.output_root.mkdir(parents=True, exist_ok=True)

        model = UNetPP3D(in_channels=1, num_classes=3, base_ch=16).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()

        patch_size = tuple(args.patch_size)
        stride = tuple(args.stride)
        dataset = TumorSegmentationDataset(
            root_dir=str(args.data_root),
            mode="val",
            patch_size=patch_size,
        )

        for index in tqdm(range(min(len(dataset), args.max_cases))):
            volume, target = dataset[index]
            prediction = sliding_window_inference(
                volume.to(device),
                model,
                patch_size=patch_size,
                stride=stride,
                device=device,
            )

            case_dir = args.output_root / f"case_{index:05d}"
            case_dir.mkdir(exist_ok=True)
            np.save(case_dir / "pred.npy", prediction.numpy())
            np.save(case_dir / "gt.npy", target.numpy())
            np.save(case_dir / "ct.npy", volume[0].numpy())

            slice_index = prediction.shape[0] // 2
            save_overlay(
                volume[0, slice_index].numpy(),
                target[slice_index].numpy(),
                prediction[slice_index].numpy(),
                case_dir / f"slice_{slice_index:03d}.png",
            )
            logger.info("Saved inference outputs for case_%05d", index)
    except Exception:
        logger.exception("UNet++ inference failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
