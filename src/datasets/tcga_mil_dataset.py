# src/datasets/tcga_mil_dataset.py

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit


CLASS_MAP = {
    "KIRC": 0,
    "KIRP": 1,
    "KICH": 2
}


class TCGAMILDataset(Dataset):

    def __init__(self, root="data/tcga_2p5d", split="train", val_ratio=0.2):

        self.root = Path(root)
        self.samples = []
        self.labels = []

        # ---- Collect all cases ----
        for cancer, label in CLASS_MAP.items():
            cancer_dir = self.root / cancer
            if not cancer_dir.exists():
                continue

            for case in cancer_dir.iterdir():
                if (case / "slices.npy").exists():
                    self.samples.append(case)
                    self.labels.append(label)

        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)

        # ---- Stratified split ----
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_ratio,
            random_state=42
        )

        train_idx, val_idx = next(splitter.split(self.samples, self.labels))

        if split == "train":
            self.samples = self.samples[train_idx]
            self.labels = self.labels[train_idx]
        else:
            self.samples = self.samples[val_idx]
            self.labels = self.labels[val_idx]

        print(f"{split.upper()} distribution:")
        unique, counts = np.unique(self.labels, return_counts=True)
        for u, c in zip(unique, counts):
            name = list(CLASS_MAP.keys())[u]
            print(f"{name}: {c}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        case_path = self.samples[idx]
        label = self.labels[idx]

        slices = np.load(case_path / "slices.npy")  # (K,H,W)
        slices = torch.tensor(slices, dtype=torch.float32)

        # ---- Safe normalization ----
        mean = slices.mean()
        std = slices.std()
        if std < 1e-6:
            std = 1.0

        slices = (slices - mean) / std

        # ---- Resize to fixed 128x128 ----
        slices = slices.unsqueeze(0)  # (1,K,H,W)
        slices = F.interpolate(
            slices,
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        )
        slices = slices.squeeze(0)  # (K,128,128)

        return slices, torch.tensor(label, dtype=torch.long)