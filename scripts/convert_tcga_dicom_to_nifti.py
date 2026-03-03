import subprocess
from pathlib import Path

RAW_ROOT = Path("data/raw/tcga_clean")
OUT_ROOT = Path("data/tcga_nifti")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

for cancer in ["KIRC", "KIRP", "KICH"]:
    cancer_dir = RAW_ROOT / cancer
    out_cancer = OUT_ROOT / cancer
    out_cancer.mkdir(exist_ok=True)

    for patient in cancer_dir.iterdir():
        dicom_dirs = list(patient.glob("**/DICOM*"))
        if not dicom_dirs:
            print(f"[SKIP] {patient.name}: no DICOM folder")
            continue

        dicom_dir = dicom_dirs[0]
        out_dir = out_cancer / patient.name
        out_dir.mkdir(exist_ok=True)

        cmd = [
            "dcm2niix",
            "-z", "y",          # gzip
            "-f", "%p",         # filename = patient ID
            "-o", str(out_dir),
            str(dicom_dir)
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"[OK] {patient.name}")
        except Exception as e:
            print(f"[FAIL] {patient.name}: {e}")