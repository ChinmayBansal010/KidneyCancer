import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.models.unetpp_3d import UNetPP3D
from src.datasets.tumor_segmentation_dataset import TumorSegmentationDataset
from src.utils.visualization import save_overlay


# -------------------------------------------------
# Sliding window inference (CORRECT)
# -------------------------------------------------
def sliding_window_inference(
    volume,
    model,
    patch_size=(96, 96, 96),
    stride=(48, 48, 48),
    num_classes=3,
    device="cuda"
):
    """
    volume: torch.Tensor [1, Z, Y, X]
    return: torch.Tensor [Z, Y, X]
    """

    model.eval()

    _, Z, Y, X = volume.shape
    pz, py, px = patch_size
    sz, sy, sx = stride

    score_map = torch.zeros((num_classes, Z, Y, X), device=device)
    count_map = torch.zeros((Z, Y, X), device=device)

    # ensure full coverage
    z_starts = list(range(0, max(Z - pz, 0) + 1, sz))
    y_starts = list(range(0, max(Y - py, 0) + 1, sy))
    x_starts = list(range(0, max(X - px, 0) + 1, sx))

    if z_starts[-1] != Z - pz:
        z_starts.append(Z - pz)
    if y_starts[-1] != Y - py:
        y_starts.append(Y - py)
    if x_starts[-1] != X - px:
        x_starts.append(X - px)

    with torch.no_grad():
        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    patch = volume[
                        :, z:z+pz, y:y+py, x:x+px
                    ].unsqueeze(0).to(device)

                    logits = model(patch)
                    probs = torch.softmax(logits, dim=1)[0]

                    score_map[
                        :, z:z+pz, y:y+py, x:x+px
                    ] += probs

                    count_map[
                        z:z+pz, y:y+py, x:x+px
                    ] += 1

    count_map = torch.clamp(count_map, min=1.0)
    score_map /= count_map.unsqueeze(0)

    return torch.argmax(score_map, dim=0).cpu()


# -------------------------------------------------
# Main
# -------------------------------------------------
def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Model (MUST MATCH TRAINING)
    model = UNetPP3D(
        in_channels=1,
        num_classes=3,
        base_ch=16
    ).to(device)

    model.load_state_dict(
        torch.load(
            "experiments/seg_unetpp/checkpoints/best_model.pth",
            map_location=device
        )
    )
    model.eval()

    ds = TumorSegmentationDataset(
        root_dir="data/tumor_roi/kits19",
        mode="val",
        patch_size=(96, 96, 96)
    )

    out_root = Path("experiments/seg_unetpp/inference_sw")
    out_root.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(min(len(ds), 20))):
        x, y = ds[idx]     # x: [1,Z,Y,X]
        volume = x.to(device)

        pred = sliding_window_inference(
            volume,
            model,
            patch_size=(96, 96, 96),
            stride=(48, 48, 48),
            device=device
        )

        case_dir = out_root / f"case_{idx:05d}"
        case_dir.mkdir(exist_ok=True)

        np.save(case_dir / "pred.npy", pred.numpy())
        np.save(case_dir / "gt.npy", y.numpy())
        np.save(case_dir / "ct.npy", x[0].numpy())

        z = pred.shape[0] // 2
        save_overlay(
            x[0, z].numpy(),
            y[z].numpy(),
            pred[z].numpy(),
            case_dir / f"slice_{z:03d}.png"
        )


if __name__ == "__main__":
    run()