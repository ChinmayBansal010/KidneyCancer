"""Run MIL classification inference on TCGA 2.5D slices."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from kidneycancer.utils.logging_utils import configure_logging


CLASS_TO_INDEX = {"KIRC": 0, "KIRP": 1, "KICH": 2}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for TCGA MIL inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("experiments/mil_b0_best.pth"))
    parser.add_argument("--data-root", type=Path, default=Path("data/tcga_2p5d"))
    parser.add_argument("--predictions-path", type=Path, default=Path("experiments/mil_predictions.csv"))
    parser.add_argument("--confusion-matrix-path", type=Path, default=Path("experiments/confusion_matrix.png"))
    parser.add_argument("--no-show", action="store_true", help="Do not open the confusion matrix plot window.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Predict TCGA classes, save probabilities, and export a confusion matrix."""
    args = parse_args(argv)
    logger = configure_logging("kidneycancer.infer_tcga_classification")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn.functional as F
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        from kidneycancer.models.mil_model import MILNet

        def load_case_tensor(case_dir: Path) -> torch.Tensor:
            slices = torch.tensor(np.load(case_dir / "slices.npy"), dtype=torch.float32)
            mean = slices.mean()
            std = slices.std()
            if std < 1e-6:
                std = torch.tensor(1.0, dtype=slices.dtype)
            slices = (slices - mean) / std
            return F.interpolate(
                slices.unsqueeze(0),
                size=(128, 128),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if not args.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint does not exist: {args.model_path}")
        if not args.data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {args.data_root}")

        args.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        args.confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = MILNet(num_classes=3, ssl_path=None).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()

        rows: list[list[float | int | str]] = []
        for cancer_dir in sorted(args.data_root.iterdir()):
            if cancer_dir.name not in CLASS_TO_INDEX:
                continue

            for case_dir in sorted(cancer_dir.iterdir()):
                slices = load_case_tensor(case_dir).to(device)
                with torch.no_grad():
                    logits, _, _ = model(slices)
                    probabilities = torch.softmax(logits, dim=0).cpu().numpy()

                rows.append(
                    [
                        case_dir.name,
                        CLASS_TO_INDEX[cancer_dir.name],
                        int(np.argmax(probabilities)),
                        float(probabilities[0]),
                        float(probabilities[1]),
                        float(probabilities[2]),
                    ]
                )

        predictions = pd.DataFrame(
            rows,
            columns=[
                "case_id",
                "true_class",
                "predicted_class",
                "prob_KIRC",
                "prob_KIRP",
                "prob_KICH",
            ],
        )
        predictions.to_csv(args.predictions_path, index=False)

        matrix = confusion_matrix(
            predictions["true_class"].values,
            predictions["predicted_class"].values,
        )
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["KIRC", "KIRP", "KICH"],
        )
        display.plot(cmap="Blues")
        plt.title("Confusion Matrix - MIL Classification")
        plt.savefig(args.confusion_matrix_path)
        if not args.no_show:
            plt.show()
        plt.close()

        logger.info("Saved predictions to %s", args.predictions_path)
        logger.info("Saved confusion matrix to %s", args.confusion_matrix_path)
        logger.info("Inference complete")
    except Exception:
        logger.exception("TCGA classification inference failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
