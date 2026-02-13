import numpy as np
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
import random

# -----------------------------
# Utility functions
# -----------------------------

def pad_if_needed(image, mask, patch_size):
    dz, dy, dx = patch_size
    iz, iy, ix = image.shape

    pad_z = max(0, dz - iz)
    pad_y = max(0, dy - iy)
    pad_x = max(0, dx - ix)

    pad_before = (
        pad_z // 2,
        pad_y // 2,
        pad_x // 2
    )
    pad_after = (
        pad_z - pad_before[0],
        pad_y - pad_before[1],
        pad_x - pad_before[2]
    )

    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        image = np.pad(
            image,
            (
                (pad_before[0], pad_after[0]),
                (pad_before[1], pad_after[1]),
                (pad_before[2], pad_after[2])
            ),
            mode="constant",
            constant_values=0.0
        )

        mask = np.pad(
            mask,
            (
                (pad_before[0], pad_after[0]),
                (pad_before[1], pad_after[1]),
                (pad_before[2], pad_after[2])
            ),
            mode="constant",
            constant_values=0
        )

    return image, mask


def extract_patch(image, mask, center, patch_size):
    dz, dy, dx = patch_size
    zc, yc, xc = center

    zmin = zc - dz // 2
    ymin = yc - dy // 2
    xmin = xc - dx // 2

    zmin = max(0, min(zmin, image.shape[0] - dz))
    ymin = max(0, min(ymin, image.shape[1] - dy))
    xmin = max(0, min(xmin, image.shape[2] - dx))

    zmax = zmin + dz
    ymax = ymin + dy
    xmax = xmin + dx

    return (
        image[zmin:zmax, ymin:ymax, xmin:xmax],
        mask[zmin:zmax, ymin:ymax, xmin:xmax]
    )


def get_centroid(mask, tumor_present):
    if tumor_present:
        coords = np.where(mask == 2)
    else:
        coords = np.where(mask > 0)

    return (
        int(coords[0].mean()),
        int(coords[1].mean()),
        int(coords[2].mean())
    )

# -----------------------------
# Dataset class
# -----------------------------

class TumorSegmentationDataset(Dataset):
    """
    Tumor segmentation dataset with:
    - B2 stochastic sampling for training
    - B1 deterministic sampling for validation
    """

    def __init__(
        self,
        root_dir: str,
        patch_size=(64, 64, 64),
        mode="train",
        samples_per_case=4,
        jitter_radius=12,
        seed=42
    ):
        """
        Args:
            root_dir: path to data/tumor_roi/kits19
            patch_size: (Z,Y,X)
            mode: 'train' or 'val'
            samples_per_case: how many samples per case (train only)
            jitter_radius: max voxel jitter (train only)
            seed: random seed
        """
        assert mode in ("train", "val")

        self.root = Path(root_dir)
        self.cases = sorted(self.root.glob("case_*"))
        assert len(self.cases) > 0, "No tumor ROI cases found"

        self.patch_size = patch_size
        self.mode = mode
        self.samples_per_case = samples_per_case
        self.jitter_radius = jitter_radius

        self.rng = random.Random(seed)

        # Validation offsets = B1 logic
        self.val_offsets = [
            (0, 0, 0),
            (8, 0, 0),
            (-8, 0, 0),
            (0, 8, 0),
            (0, -8, 0),
        ]

    def __len__(self):
        if self.mode == "train":
            return len(self.cases) * self.samples_per_case
        else:
            return len(self.cases) * len(self.val_offsets)

    def __getitem__(self, idx):
        if self.mode == "train":
            case_idx = idx // self.samples_per_case
        else:
            case_idx = idx // len(self.val_offsets)

        case_dir = self.cases[case_idx]

        data = np.load(case_dir / "tumor_data.npz")
        image = data["image"].astype(np.float32)
        mask = data["mask"].astype(np.uint8)

        with open(case_dir / "tumor_meta.json", "r") as f:
            meta = json.load(f)

        tumor_present = meta["tumor_present"]

        image, mask = pad_if_needed(image, mask, self.patch_size)

        center = get_centroid(mask, tumor_present)

        if self.mode == "train":
            # -------- B2 stochastic sampling --------
            dz = self.rng.randint(-self.jitter_radius, self.jitter_radius)
            dy = self.rng.randint(-self.jitter_radius, self.jitter_radius)
            dx = self.rng.randint(-self.jitter_radius, self.jitter_radius)

            center = (
                center[0] + dz,
                center[1] + dy,
                center[2] + dx
            )

        else:
            # -------- B1 deterministic sampling --------
            offset_idx = idx % len(self.val_offsets)
            oz, oy, ox = self.val_offsets[offset_idx]

            center = (
                center[0] + oz,
                center[1] + oy,
                center[2] + ox
            )

        patch_img, patch_mask = extract_patch(
            image, mask, center, self.patch_size
        )

        # Torch tensors
        patch_img = torch.from_numpy(patch_img).unsqueeze(0)  # [1,Z,Y,X]
        patch_mask = torch.from_numpy(patch_mask).long()      # [Z,Y,X]

        return patch_img, patch_mask
