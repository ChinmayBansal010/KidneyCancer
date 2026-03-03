# scripts/infer_tcga_classification.py

import torch
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from src.models.mil_model import MILNet

MODEL_PATH = "experiments/mil_b0_best.pth"
DATA_ROOT = Path("data/tcga_2p5d")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = MILNet(num_classes=3, ssl_path=None).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

results = []

for cancer_dir in DATA_ROOT.iterdir():
    for case in cancer_dir.iterdir():

        slices = np.load(case / "slices.npy")
        slices = torch.tensor(slices, dtype=torch.float32)

        mean = slices.mean()
        std = slices.std()
        if std < 1e-6:
            std = 1.0

        slices = (slices - mean) / std

        slices = torch.nn.functional.interpolate(
            slices.unsqueeze(0),
            size=(128,128),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        slices = slices.to(device)

        with torch.no_grad():
            logits, _, _ = model(slices)
            probs = torch.softmax(logits, dim=0).cpu().numpy()

        pred = np.argmax(probs)

        true_label = {"KIRC":0, "KIRP":1, "KICH":2}[cancer_dir.name]

        results.append([
            case.name,
            true_label,
            pred,
            probs[0],
            probs[1],
            probs[2]
        ])

df = pd.DataFrame(results, columns=[
    "case_id",
    "true_class",
    "predicted_class",
    "prob_KIRC",
    "prob_KIRP",
    "prob_KICH"
])

df.to_csv("experiments/mil_predictions.csv", index=False)

y_true = df["true_class"].values
y_pred = df["predicted_class"].values

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["KIRC", "KIRP", "KICH"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - MIL Classification")
plt.savefig("experiments/confusion_matrix.png")
plt.show()

print("Inference complete.")