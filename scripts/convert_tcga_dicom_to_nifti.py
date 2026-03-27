import subprocess
from pathlib import Path

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from kidneycancer.utils.logging_utils import configure_logging

RAW_ROOT = Path("data/raw/tcga_clean")
OUT_ROOT = Path("data/tcga_nifti")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

LOGGER = configure_logging("kidneycancer.convert_tcga_dicom_to_nifti")


def main() -> None:
    """Convert TCGA DICOM studies to gzipped NIfTI files."""
    for cancer in ["KIRC", "KIRP", "KICH"]:
        cancer_dir = RAW_ROOT / cancer
        out_cancer = OUT_ROOT / cancer
        out_cancer.mkdir(exist_ok=True)

        for patient in cancer_dir.iterdir():
            dicom_dirs = list(patient.glob("**/DICOM*"))
            if not dicom_dirs:
                LOGGER.warning("Skipping %s: no DICOM folder found", patient.name)
                continue

            dicom_dir = dicom_dirs[0]
            out_dir = out_cancer / patient.name
            out_dir.mkdir(exist_ok=True)

            cmd = [
                "dcm2niix",
                "-z", "y",
                "-f", "%p",
                "-o", str(out_dir),
                str(dicom_dir),
            ]

            try:
                subprocess.run(cmd, check=True)
                LOGGER.info("Converted %s", patient.name)
            except Exception as exc:
                LOGGER.error("Failed to convert %s: %s", patient.name, exc)


if __name__ == "__main__":
    main()
