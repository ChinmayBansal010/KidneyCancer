# KidneyCancerDetection

KidneyCancerDetection is a Python project for kidney tumor preprocessing, localization, segmentation, and TCGA-based classification experiments.

The repository now uses a professional `src/` package layout, exposes a package CLI, and keeps experiment-oriented scripts in `scripts/`.

## What This Repository Contains

- A reusable package under `src/kidneycancer/`
- A package CLI for the main KiTS19 preprocessing pipeline
- Training and inference scripts for segmentation and MIL experiments
- Utility scripts for TCGA conversion and preparation

## Requirements

- Python `3.10+`
- Windows, Linux, or WSL with Python and `pip`
- Enough disk space for medical imaging data and experiment outputs

Optional external tools:

- `aria2c` for high-speed KiTS19 downloads via `scripts/download.py`
- `dcm2niix` for TCGA DICOM to NIfTI conversion via `scripts/convert_tcga_dicom_to_nifti.py`

## Installation

### PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

### Bash

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

## Verify The Install

```powershell
python -m kidneycancer --help
```

Or, after installation:

```powershell
kidneycancer --help
```

## Main CLI Commands

The package CLI is intended for the core KiTS19 data pipeline:

```powershell
kidneycancer preprocess
kidneycancer localize-kidneys
kidneycancer localize-tumors
kidneycancer extract-patches
kidneycancer build-2p5d
```

These commands currently read their paths and constants from the config modules in:

- `src/kidneycancer/preprocessing/config.py`
- `src/kidneycancer/kidney_localization/config.py`
- `src/kidneycancer/tumor_localization/config.py`
- `src/kidneycancer/patch_extraction/config.py`
- `src/kidneycancer/slicing_2p5d/config.py`

## Repository Layout

- `src/kidneycancer/`: installable source package
- `scripts/`: runnable experiment and utility scripts
- `data/`: local datasets and generated intermediate data
- `experiments/`: checkpoints, logs, predictions, and qualitative outputs
- `docs/`: user-facing documentation

## Documentation

For full setup instructions, data layout expectations, workflow order, and a command-by-command reference for every CLI command and script, see:

- [docs/USER_GUIDE.md](docs/USER_GUIDE.md)

## Notes

- The CLI commands are the safest entry point for the main KiTS19 preparation pipeline.
- Many scripts in `scripts/` are experiment-oriented and use hard-coded paths. Review them before changing data locations.
- Root-level wrapper scripts such as `scripts/run_preprocessing.py` are kept for convenience, but the package CLI is the preferred interface.
