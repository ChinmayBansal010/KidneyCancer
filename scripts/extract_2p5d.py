import numpy as np
from pathlib import Path
from scipy.stats import entropy

TOP_K = 5
IN_ROOT = Path("data/tcga_roi")
OUT_ROOT = Path("data/tcga_2p5d")
OUT_ROOT.mkdir(exist_ok=True)

def H(slice_):
    hist, _ = np.histogram(slice_, bins=256, density=True)
    hist = hist[hist > 0]
    return entropy(hist, base=2)

for cancer_type in IN_ROOT.iterdir():

    for case in cancer_type.iterdir():

        ct_path = case / "ct.npy"
        if not ct_path.exists():
            continue

        ct = np.load(ct_path)  # (Z, H, W)

        # Skip garbage ROIs
        if ct.ndim != 3:
            continue

        Z, H_, W_ = ct.shape

        if H_ < 32 or W_ < 32:
            print(f"[SKIP SMALL ROI] {case.name} shape={ct.shape}")
            continue

        scores = [(i, H(ct[i])) for i in range(Z)]
        top = sorted(scores, key=lambda x: x[1], reverse=True)[:TOP_K]

        # stack as channels
        slices = np.stack([ct[i] for i,_ in top], axis=0)  # (K, H, W)

        out = OUT_ROOT / cancer_type.name / case.name
        out.mkdir(parents=True, exist_ok=True)

        np.save(out / "slices.npy", slices.astype(np.float32))