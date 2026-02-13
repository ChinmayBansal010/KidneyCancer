import json
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset


class MILSliceDataset(Dataset):
    """
    MIL Dataset:
    - One item = one case (bag)
    - Bag = [axial, coronal, sagittal] slices
    """

    def __init__(
        self,
        root_dir: str,
        label_source: str = "kits19",
        label_csv: str | None = None
    ):
        """
        root_dir: data/slices_2p5d/kits19
        label_source: 'kits19' or 'tcga'
        label_csv: path to CSV for TCGA (ignored for KiTS19)
        """
        self.root = Path(root_dir)
        self.cases = sorted(self.root.glob("case_*"))
        assert len(self.cases) > 0, "No MIL cases found"

        self.label_source = label_source

        # TCGA placeholder (no refactor later)
        self.tcga_labels = {}
        if label_source == "tcga":
            assert label_csv is not None
            self._load_tcga_labels(label_csv)

    def _load_tcga_labels(self, csv_path):
        # Placeholder for later
        # case_id -> label
        raise NotImplementedError("TCGA label loading will be added later")

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_dir = self.cases[idx]

        data = np.load(case_dir / "slice_2p5d.npz")
        images = data["image"]    # [3, H, W]
        masks = data["mask"]

        with open(case_dir / "slice_meta.json") as f:
            meta = json.load(f)

        # ----- label handling -----
        if self.label_source == "kits19":
            label = int(meta["tumor_present"])
        else:
            label = self.tcga_labels[case_dir.name]

        # Torch tensors
        images = torch.from_numpy(images).float()   # [3, H, W]
        images = images.unsqueeze(1)                # [3, 1, H, W]

        label = torch.tensor(label, dtype=torch.long)

        return images, label, case_dir.name
