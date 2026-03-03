import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

from src.models.unetpp_3d import UNetPP3D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "experiments/seg_unetpp/checkpoints/best_model.pth"
IN_ROOT = Path("data/tcga_nifti")
OUT_ROOT = Path("data/tcga_masks")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

PATCH = (96, 96, 96)
STRIDE = (48, 48, 48)
NUM_CLASSES = 3


def sliding_window(volume, model):
    """
    volume: np.ndarray [Z,Y,X]
    returns: np.ndarray [Z,Y,X]
    """
    Z, Y, X = volume.shape
    pz, py, px = PATCH
    sz, sy, sx = STRIDE

    score = np.zeros((NUM_CLASSES, Z, Y, X), dtype=np.float32)
    count = np.zeros((Z, Y, X), dtype=np.float32)

    for z in range(0, Z - pz + 1, sz):
        for y in range(0, Y - py + 1, sy):
            for x in range(0, X - px + 1, sx):
                patch = volume[z:z+pz, y:y+py, x:x+px]

                x_t = torch.from_numpy(patch)\
                          .unsqueeze(0)\
                          .unsqueeze(0)\
                          .to(DEVICE)

                with torch.no_grad():
                    logits = model(x_t)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                score[:, z:z+pz, y:y+py, x:x+px] += probs
                count[z:z+pz, y:y+py, x:x+px] += 1

    score /= np.maximum(count, 1e-6)
    return score.argmax(axis=0).astype(np.uint8)

def normalize_tcga_volume(img):
    vol = img.get_fdata()

    # Remove singleton dims
    vol = np.squeeze(vol)

    # If still 4D → keep first channel only
    if vol.ndim == 4:
        vol = vol[..., 0]

    # After squeeze, valid cases should be 3D
    if vol.ndim != 3:
        return None

    # Skip useless scans (1 slice scout)
    if vol.shape[2] <= 5:
        return None

    # If stored as (H,W,Z) → transpose to (Z,H,W)
    if vol.shape[0] in [512, 600] and vol.shape[1] in [512, 600]:
        vol = np.transpose(vol, (2, 0, 1))

    return vol.astype(np.float32)

# ----------------------------
# MAIN
# ----------------------------
model = UNetPP3D(in_channels=1, num_classes=3, base_ch=16).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

all_files = list(IN_ROOT.rglob("*.nii.gz"))

todo = []
for nii in all_files:
    out_dir = OUT_ROOT / nii.parent.parent.name
    out_file = out_dir / nii.name
    if not out_file.exists():
        todo.append(nii)

print("\n----------------------------")
print(f"Total scans       : {len(all_files)}")
print(f"Already processed : {len(all_files) - len(todo)}")
print(f"Remaining         : {len(todo)}")
print("----------------------------\n")

for nii in tqdm(todo):

    out_dir = OUT_ROOT / nii.parent.parent.name
    out_dir.mkdir(exist_ok=True)

    img = nib.load(nii)
    vol = normalize_tcga_volume(img)

    if vol is None:
        print(f"[SKIP INVALID] {nii.name}")
        continue

    vol = (vol - np.mean(vol)) / (np.std(vol) + 1e-5)

    pred = sliding_window(vol, model)

    nib.save(
        nib.Nifti1Image(pred, img.affine),
        out_dir / nii.name
    )