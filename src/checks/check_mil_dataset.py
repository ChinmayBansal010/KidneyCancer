from pathlib import Path
from src.mil.dataset import MILSliceDataset

def check_mil_dataset():
    ds = MILSliceDataset(
        root_dir="data/slices_2p5d/kits19",
        label_source="kits19"
    )

    assert len(ds) > 0

    images, label, case_id = ds[0]

    assert images.shape[0] == 3        # instances
    assert images.shape[1] == 1        # channel
    assert images.ndim == 4             # [3,1,H,W]
    assert label.item() in (0, 1)

    print("✓ MIL dataset sanity passed")

if __name__ == "__main__":
    check_mil_dataset()
