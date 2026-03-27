import shutil
from pathlib import Path
from collections import defaultdict
import logging

import pydicom
from tqdm import tqdm

from kidneycancer.utils.logging_utils import configure_logging

# ============================================================
# CONFIG
# ============================================================
RAW_TCGA_ROOT = Path("data/raw/tcga")
OUTPUT_ROOT = Path("data/raw/tcga_clean")

SUBTYPES = ["TCGA-KIRC", "TCGA-KIRP", "TCGA-KICH"]

VALID_MODALITY = "CT"
MIN_SLICES = 80
LOGGER = logging.getLogger(__name__)


# ============================================================
# UTILS
# ============================================================
def load_dicom_safe(path):
    try:
        return pydicom.dcmread(path, stop_before_pixels=True)
    except Exception:
        return None


def is_scout(ds):
    desc = getattr(ds, "SeriesDescription", "")
    desc = desc.upper()
    return any(k in desc for k in ["SCOUT", "LOCALIZER", "APLAT"])


def has_consistent_spacing(files):
    spacings = set()
    for f in files[:10]:  # sample only
        ds = load_dicom_safe(f)
        if ds is None:
            return False
        try:
            spacing = (
                float(ds.PixelSpacing[0]),
                float(ds.PixelSpacing[1]),
                float(ds.SliceThickness),
            )
            spacings.add(spacing)
        except Exception:
            return False
    return len(spacings) == 1


def score_series(files):
    ds = load_dicom_safe(files[0])
    slice_count = len(files)
    try:
        thickness = float(ds.SliceThickness)
    except Exception:
        thickness = 999.0

    # Heuristic: prefer many slices + thinner slices
    return slice_count * 10 - thickness


def find_subtype_roots(root, subtype):
    """
    Find all manifest folders containing a given subtype.
    """
    roots = []
    for manifest in root.glob("manifest-*"):
        candidate = manifest / subtype
        if candidate.exists():
            roots.append(candidate)
    return roots


# ============================================================
# CORE LOGIC
# ============================================================
def process_patient(patient_dir, out_dir):
    dicom_files = list(patient_dir.rglob("*.dcm"))
    if not dicom_files:
        return False, "no_dicom"

    # study_uid -> series_uid -> files
    study_map = defaultdict(lambda: defaultdict(list))

    for dcm_path in dicom_files:
        ds = load_dicom_safe(dcm_path)
        if ds is None:
            continue

        if ds.Modality != VALID_MODALITY:
            continue

        if is_scout(ds):
            continue

        study_uid = getattr(ds, "StudyInstanceUID", None)
        series_uid = getattr(ds, "SeriesInstanceUID", None)

        if study_uid is None or series_uid is None:
            continue

        study_map[study_uid][series_uid].append(dcm_path)

    best_files = None
    best_score = -1

    for study_uid, series_dict in study_map.items():
        for series_uid, files in series_dict.items():
            if len(files) < MIN_SLICES:
                continue
            if not has_consistent_spacing(files):
                continue

            score = score_series(files)
            if score > best_score:
                best_score = score
                best_files = files

    if best_files is None:
        return False, "no_valid_series"

    out_series_dir = out_dir / "DICOM" / "series_1"
    out_series_dir.mkdir(parents=True, exist_ok=True)

    for src in best_files:
        shutil.copy2(src, out_series_dir / src.name)

    return True, f"slices={len(best_files)}"


# ============================================================
# DRIVER
# ============================================================
def main():
    configure_logging("kidneycancer.select_best_ct_series")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log = []

    for subtype in SUBTYPES:
        LOGGER.info("Processing %s", subtype)
        subtype_roots = find_subtype_roots(RAW_TCGA_ROOT, subtype)

        if not subtype_roots:
            LOGGER.warning("Skipping %s because it was not found in any manifest", subtype)
            continue

        out_subtype_root = OUTPUT_ROOT / subtype.replace("TCGA-", "")
        out_subtype_root.mkdir(exist_ok=True)

        for subtype_root in subtype_roots:
            patient_dirs = [
                p for p in subtype_root.iterdir()
                if p.is_dir() and p.name.startswith("TCGA")
            ]

            for patient_dir in tqdm(patient_dirs, desc=subtype):
                patient_id = patient_dir.name
                out_patient_dir = out_subtype_root / patient_id

                success, msg = process_patient(patient_dir, out_patient_dir)
                log.append((subtype, patient_id, success, msg))

    log_path = OUTPUT_ROOT / "series_selection_log.txt"
    with open(log_path, "w") as f:
        for row in log:
            f.write(",".join(map(str, row)) + "\n")

    LOGGER.info("Done. Log written to %s", log_path)


if __name__ == "__main__":
    main()
