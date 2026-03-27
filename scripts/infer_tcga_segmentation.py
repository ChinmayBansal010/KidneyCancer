"""Run TCGA segmentation inference with a trained UNet++ checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from kidneycancer.utils.logging_utils import configure_logging


NUM_CLASSES = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for TCGA segmentation inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("experiments/seg_unetpp/checkpoints/best_model.pth"))
    parser.add_argument("--input-root", type=Path, default=Path("data/tcga_nifti"))
    parser.add_argument("--output-root", type=Path, default=Path("data/tcga_masks"))
    parser.add_argument("--patch-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--stride", type=int, nargs=3, default=(48, 48, 48))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Segment all unprocessed TCGA scans and save the predicted masks."""
    args = parse_args(argv)
    logger = configure_logging("kidneycancer.infer_tcga_segmentation")

    try:
        import nibabel as nib
        import numpy as np
        import torch
        from tqdm import tqdm

        from kidneycancer.models.unetpp_3d import UNetPP3D

        def sliding_window(
            volume: np.ndarray,
            model: torch.nn.Module,
            device: str,
            patch_size: tuple[int, int, int],
            stride: tuple[int, int, int],
        ) -> np.ndarray:
            depth, height, width = volume.shape
            patch_depth, patch_height, patch_width = patch_size
            stride_depth, stride_height, stride_width = stride

            score_map = np.zeros((NUM_CLASSES, depth, height, width), dtype=np.float32)
            count_map = np.zeros((depth, height, width), dtype=np.float32)

            for z_index in range(0, depth - patch_depth + 1, stride_depth):
                for y_index in range(0, height - patch_height + 1, stride_height):
                    for x_index in range(0, width - patch_width + 1, stride_width):
                        patch = volume[
                            z_index : z_index + patch_depth,
                            y_index : y_index + patch_height,
                            x_index : x_index + patch_width,
                        ]
                        patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)

                        with torch.no_grad():
                            logits = model(patch_tensor)
                            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

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

            score_map /= np.maximum(count_map, 1e-6)
            return score_map.argmax(axis=0).astype(np.uint8)

        def normalize_tcga_volume(image: nib.Nifti1Image) -> np.ndarray | None:
            volume = np.squeeze(image.get_fdata())
            if volume.ndim == 4:
                volume = volume[..., 0]
            if volume.ndim != 3:
                return None
            if volume.shape[2] <= 5:
                return None
            if volume.shape[0] in [512, 600] and volume.shape[1] in [512, 600]:
                volume = np.transpose(volume, (2, 0, 1))
            return volume.astype(np.float32)

        if not args.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint does not exist: {args.model_path}")
        if not args.input_root.exists():
            raise FileNotFoundError(f"Input root does not exist: {args.input_root}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        args.output_root.mkdir(parents=True, exist_ok=True)
        model = UNetPP3D(in_channels=1, num_classes=3, base_ch=16).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()

        patch_size = tuple(args.patch_size)
        stride = tuple(args.stride)
        all_files = list(args.input_root.rglob("*.nii.gz"))
        todo = []
        for nii_path in all_files:
            case_output_dir = args.output_root / nii_path.parent.parent.name
            output_file = case_output_dir / nii_path.name
            if not output_file.exists():
                todo.append(nii_path)

        logger.info(
            "Found %s scans | already_processed=%s | remaining=%s",
            len(all_files),
            len(all_files) - len(todo),
            len(todo),
        )

        for nii_path in tqdm(todo):
            case_output_dir = args.output_root / nii_path.parent.parent.name
            case_output_dir.mkdir(exist_ok=True)

            image = nib.load(nii_path)
            volume = normalize_tcga_volume(image)
            if volume is None:
                logger.warning("Skipping invalid scan: %s", nii_path.name)
                continue

            volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-5)
            prediction = sliding_window(volume, model, device, patch_size, stride)
            nib.save(nib.Nifti1Image(prediction, image.affine), case_output_dir / nii_path.name)
            logger.info("Saved mask for %s", nii_path.name)
    except Exception:
        logger.exception("TCGA segmentation inference failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
