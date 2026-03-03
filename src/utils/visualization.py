import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt


def extract_boundary(mask):
    """Binary boundary extraction"""
    mask = mask.astype(bool)
    eroded = np.logical_and(
        np.roll(mask, 1, axis=0),
        np.roll(mask, -1, axis=0)
    )
    return mask ^ eroded


def boundary_distance_map(gt_mask):
    """
    Distance map used for HD95 visualization
    """
    boundary = extract_boundary(gt_mask)
    return distance_transform_edt(~boundary)


def save_boundary_overlay(
    ct_slice,
    gt_mask,
    pred_mask,
    dice,
    iou,
    hd95,
    out_path
):
    """
    Creates a 4-panel qualitative figure
    """

    gt_boundary = extract_boundary(gt_mask)
    pred_boundary = extract_boundary(pred_mask)

    dist_map = boundary_distance_map(gt_mask)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # CT
    axes[0].imshow(ct_slice, cmap="gray")
    axes[0].set_title("CT Slice")
    axes[0].axis("off")

    # GT vs Pred Boundary
    axes[1].imshow(ct_slice, cmap="gray")
    axes[1].contour(gt_boundary, colors="green", linewidths=1)
    axes[1].contour(pred_boundary, colors="red", linewidths=1)
    axes[1].set_title("GT (Green) vs Pred (Red)")
    axes[1].axis("off")

    # Boundary Error Heatmap
    axes[2].imshow(dist_map, cmap="hot")
    axes[2].set_title("Boundary Distance Map")
    axes[2].axis("off")

    # Metrics Panel
    axes[3].axis("off")
    axes[3].text(
        0.05, 0.6,
        f"Dice  : {dice:.4f}\n"
        f"IoU   : {iou:.4f}\n"
        f"HD95  : {hd95:.2f}",
        fontsize=12
    )
    axes[3].set_title("Metrics")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_overlay(ct, gt, pred, out_path):
    """
    Simple qualitative overlay:
    - CT slice
    - GT mask overlay
    - Pred mask overlay
    """

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(ct, cmap="gray")
    plt.title("CT")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(ct, cmap="gray")
    plt.imshow(gt, alpha=0.5, cmap="jet")
    plt.title("GT Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(ct, cmap="gray")
    plt.imshow(pred, alpha=0.5, cmap="jet")
    plt.title("Pred Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()