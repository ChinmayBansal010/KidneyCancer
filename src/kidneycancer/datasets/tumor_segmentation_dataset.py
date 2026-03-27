import numpy as np
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
import random

TUMOR_LABEL = 2
KIDNEY_LABEL = 1


def pad_if_needed(image, mask, patch_size):
    """
    Pad image and mask if needed to match patch size.

    Args:
        image (numpy.ndarray): 3D image.
        mask (numpy.ndarray): 3D mask.
        patch_size (tuple): (dz, dy, dx)

    Returns:
        tuple: (padded_image, padded_mask)
    """
    dz, dy, dx = patch_size
    iz, iy, ix = image.shape
    pad_z = max(0, dz - iz)
    pad_y = max(0, dy - iy)
    pad_x = max(0, dx - ix)
    pad_before = (pad_z // 2, pad_y // 2, pad_x // 2)
    pad_after = (pad_z - pad_before[0], pad_y - pad_before[1], pad_x - pad_before[2])
    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        image = np.pad(
            image,
            (
                (pad_before[0], pad_after[0]),
                (pad_before[1], pad_after[1]),
                (pad_before[2], pad_after[2]),
            ),
            mode="constant",
            constant_values=0.0,
        )
        mask = np.pad(
            mask,
            (
                (pad_before[0], pad_after[0]),
                (pad_before[1], pad_after[1]),
                (pad_before[2], pad_after[2]),
            ),
            mode="constant",
            constant_values=0,
        )
    return image, mask


def extract_patch(image, mask, center, patch_size):
    """
    Extract a patch from an image and mask given a center and patch size.

    Args:
        image (numpy.ndarray): 3D image.
        mask (numpy.ndarray): 3D mask.
        center (tuple): (z, y, x)
        patch_size (tuple): (dz, dy, dx)

    Returns:
        tuple: (patched_image, patched_mask)

    Notes:
        The patch is extracted from the image and mask by taking the center as the
        center of the patch. If the patch goes outside the image boundaries, it
        is clamped to the nearest boundary.
    """
    dz, dy, dx = patch_size
    zc, yc, xc = center
    zmin = max(0, min(zc - dz // 2, image.shape[0] - dz))
    ymin = max(0, min(yc - dy // 2, image.shape[1] - dy))
    xmin = max(0, min(xc - dx // 2, image.shape[2] - dx))
    zmax, ymax, xmax = zmin + dz, ymin + dy, xmin + dx
    return (
        image[zmin:zmax, ymin:ymax, xmin:xmax].copy(),
        mask[zmin:zmax, ymin:ymax, xmin:xmax].copy(),
    )


def get_centroid(mask, tumor_present):
    """
    Get the centroid of the tumor/kidney voxels in the given mask.

    Args:
        mask (numpy.ndarray): 3D mask.
        tumor_present (bool): True if tumor voxels are present, False otherwise.

    Returns:
        tuple: (z, y, x) coordinates of the centroid.

    Notes:
        If tumor voxels are present, the centroid of the tumor voxels is returned.
        If tumor voxels are not present, the centroid of the kidney voxels is returned.
        If no voxels are present in the mask, an empty tuple is returned.
    """
    if tumor_present:
        coords = np.where(mask == TUMOR_LABEL)
        if len(coords[0]) == 0:
            coords = np.where(mask > 0)
    else:
        coords = np.where(mask > 0)
    return (int(coords[0].mean()), int(coords[1].mean()), int(coords[2].mean()))


class TumorSegmentationDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        patch_size=(64, 64, 64),
        mode="train",
        samples_per_case=4,
        jitter_radius=12,
        seed=42,
    ):
        """
        Initialize the TumorSegmentationDataset.

        Args:
            root_dir (str): Path to the dataset root directory.
            patch_size (tuple): Patch size (dz, dy, dx).
            mode (str): Mode of the dataset. Can be either "train" or "val".
            samples_per_case (int): Number of samples to extract from each case in the dataset.
            jitter_radius (int): Maximum jitter radius for training samples.
            seed (int): Random seed for reproducibility.

        Notes:
            The dataset root directory should contain subdirectories named "case_*".
            Each case directory should contain a "tumor_data.npz" file containing the image and mask data.
            The "tumor_meta.json" file contains metadata about the tumor, including the presence and shape of the tumor.
        """
        assert mode in ("train", "val")
        self.root = Path(root_dir)
        self.cases = sorted(self.root.glob("case_*"))
        self.patch_size = patch_size
        self.mode = mode
        self.samples_per_case = samples_per_case
        self.jitter_radius = jitter_radius
        self.rng = random.Random(seed)
        self.val_offsets = [(0, 0, 0), (8, 0, 0), (-8, 0, 0), (0, 8, 0), (0, -8, 0)]

    def __len__(self):
        """
        Returns the length of the dataset.

        If the dataset is in "train" mode, the length is the number of cases multiplied by the number of samples per case.
        If the dataset is in "val" mode, the length is the number of cases multiplied by the number of validation offsets.
        """
        if self.mode == "train":
            return len(self.cases) * self.samples_per_case
        return len(self.cases) * len(self.val_offsets)

    def __getitem__(self, idx):
        if self.mode == "train":
            case_idx = idx // self.samples_per_case
        else:
            case_idx = idx // len(self.val_offsets)

        case_dir = self.cases[case_idx]
        with np.load(case_dir / "tumor_data.npz") as data:
            image = data["image"].astype(np.float32)
            mask = data["mask"].astype(np.uint8)

        with open(case_dir / "tumor_meta.json", "r") as f:
            meta = json.load(f)

        image, mask = pad_if_needed(image, mask, self.patch_size)
        center = get_centroid(mask, meta["tumor_present"])

        if self.mode == "train":
            dz = self.rng.randint(-self.jitter_radius, self.jitter_radius)
            dy = self.rng.randint(-self.jitter_radius, self.jitter_radius)
            dx = self.rng.randint(-self.jitter_radius, self.jitter_radius)
            center = (center[0] + dz, center[1] + dy, center[2] + dx)
        else:
            oz, oy, ox = self.val_offsets[idx % len(self.val_offsets)]
            center = (center[0] + oz, center[1] + oy, center[2] + ox)

        center = (
            max(0, min(center[0], image.shape[0] - 1)),
            max(0, min(center[1], image.shape[1] - 1)),
            max(0, min(center[2], image.shape[2] - 1)),
        )

        patch_img, patch_mask = extract_patch(image, mask, center, self.patch_size)

        del image, mask

        return (
            torch.from_numpy(patch_img).unsqueeze(0),
            torch.from_numpy(patch_mask).long(),
        )
