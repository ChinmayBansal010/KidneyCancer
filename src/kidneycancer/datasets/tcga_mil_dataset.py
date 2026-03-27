"""Datasets for TCGA MIL classification experiments."""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset


CLASS_MAP = {"KIRC": 0, "KIRP": 1, "KICH": 2}
LOGGER = logging.getLogger(__name__)


class TCGAMILDataset(Dataset):
    """Load TCGA 2.5D slices and produce a stratified train or validation split."""

    def __init__(self, root: str = "data/tcga_2p5d", split: str = "train", val_ratio: float = 0.2):
        self.root = Path(root)
        self.samples: np.ndarray
        self.labels: np.ndarray

        sample_paths: list[Path] = []
        sample_labels: list[int] = []
        for cancer_name, label in CLASS_MAP.items():
            cancer_dir = self.root / cancer_name
            if not cancer_dir.exists():
                continue
            for case_dir in sorted(cancer_dir.iterdir()):
                if (case_dir / "slices.npy").exists():
                    sample_paths.append(case_dir)
                    sample_labels.append(label)

        self.samples = np.array(sample_paths)
        self.labels = np.array(sample_labels)

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_ratio,
            random_state=42,
        )
        train_idx, val_idx = next(splitter.split(self.samples, self.labels))

        if split == "train":
            self.samples = self.samples[train_idx]
            self.labels = self.labels[train_idx]
        elif split == "val":
            self.samples = self.samples[val_idx]
            self.labels = self.labels[val_idx]
        else:
            raise ValueError(f"Unsupported split: {split}")

        LOGGER.info("%s distribution:", split.upper())
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            class_name = list(CLASS_MAP.keys())[label]
            LOGGER.info("%s: %s", class_name, count)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        case_path = self.samples[idx]
        label = self.labels[idx]

        slices = torch.tensor(np.load(case_path / "slices.npy"), dtype=torch.float32)
        mean = slices.mean()
        std = slices.std()
        if std < 1e-6:
            std = torch.tensor(1.0, dtype=slices.dtype)

        slices = (slices - mean) / std
        slices = F.interpolate(
            slices.unsqueeze(0),
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return slices, torch.tensor(label, dtype=torch.long)
