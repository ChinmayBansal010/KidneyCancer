# User Guide

## Overview

This guide explains how to install the project, what directory layout it expects, and how to run every user-facing command currently present in the repository.

There are two ways to run code in this repository:

1. Use the package CLI for the main KiTS19 preprocessing pipeline
2. Run individual scripts from `scripts/` for experiments, inference, or utility tasks

## 1. Installation

### Python Version

This project requires Python `3.10+`.

### Create A Virtual Environment

PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

Bash:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

### What `pip install -e .` Does

- Installs the `kidneycancer` package in editable mode
- Makes `python -m kidneycancer ...` available
- Makes the console command `kidneycancer` available
- Installs Python dependencies from `pyproject.toml`

### Optional External Tools

Some scripts rely on tools outside Python:

- `aria2c`
  Used by `scripts/download.py`
  Purpose: fast download of KiTS19 imaging volumes

- `dcm2niix`
  Used by `scripts/convert_tcga_dicom_to_nifti.py`
  Purpose: convert TCGA DICOM studies to NIfTI

## 2. Verify The Installation

```powershell
python -m kidneycancer --help
```

Expected result:

- Help text listing the available CLI subcommands

## 3. Expected Directory Layout

The code assumes a repo-local data layout under `data/`.

### KiTS19 Data Flow

```text
data/raw/kits19
data/preprocessed/kits19
data/kidney_roi/kits19
data/tumor_roi/kits19
data/patches/kits19
data/slices_2p5d/kits19
```

### TCGA Data Flow

```text
data/raw/tcga_clean
data/tcga_nifti
data/tcga_masks
data/tcga_masks_lcc
data/tcga_roi
data/tcga_2p5d
```

### Experiment Outputs

```text
experiments/
checkpoints/
```

Different scripts write different outputs, including:

- segmentation checkpoints
- MIL checkpoints
- predictions
- qualitative overlays
- logs

## 4. Main KiTS19 Pipeline

The recommended processing order for KiTS19 is:

1. Download or place raw KiTS19 volumes in `data/raw/kits19`
2. Preprocess the raw data
3. Localize kidneys
4. Localize tumors
5. Extract patches or build 2.5D slices

### Preferred Interface

Use the package CLI:

```powershell
kidneycancer preprocess
kidneycancer localize-kidneys
kidneycancer localize-tumors
kidneycancer extract-patches
kidneycancer build-2p5d
```

### Equivalent `python -m` Usage

```powershell
python -m kidneycancer preprocess
python -m kidneycancer localize-kidneys
python -m kidneycancer localize-tumors
python -m kidneycancer extract-patches
python -m kidneycancer build-2p5d
```

## 5. CLI Command Reference

These commands are defined in `src/kidneycancer/cli.py`.

### `kidneycancer preprocess`

Purpose:

- Reads raw KiTS19 cases from `data/raw/kits19`
- Resamples, normalizes, and crops each case
- Writes outputs to `data/preprocessed/kits19`

Config source:

- `src/kidneycancer/preprocessing/config.py`

Important defaults:

- `RAW_ROOT = "data/raw/kits19"`
- `OUT_ROOT = "data/preprocessed/kits19"`
- `TARGET_SPACING = (1.0, 1.0, 1.0)`
- `HU_MIN = -150`
- `HU_MAX = 250`

Example:

```powershell
kidneycancer preprocess
```

### `kidneycancer localize-kidneys`

Purpose:

- Reads preprocessed KiTS19 cases
- Extracts kidney-focused crops from the volume
- Writes outputs to `data/kidney_roi/kits19`

Config source:

- `src/kidneycancer/kidney_localization/config.py`

Important defaults:

- `PREPROCESSED_ROOT = "data/preprocessed/kits19"`
- `OUT_ROOT = "data/kidney_roi/kits19"`

Example:

```powershell
kidneycancer localize-kidneys
```

### `kidneycancer localize-tumors`

Purpose:

- Reads kidney-localized data
- Crops around tumor regions when present
- Writes outputs to `data/tumor_roi/kits19`

Config source:

- `src/kidneycancer/tumor_localization/config.py`

Important defaults:

- `KIDNEY_ROI_ROOT = "data/kidney_roi/kits19"`
- `OUT_ROOT = "data/tumor_roi/kits19"`

Example:

```powershell
kidneycancer localize-tumors
```

### `kidneycancer extract-patches`

Purpose:

- Reads tumor-localized KiTS19 cases
- Extracts deterministic 3D patches
- Writes outputs to `data/patches/kits19`

Config source:

- `src/kidneycancer/patch_extraction/config.py`

Important defaults:

- `TUMOR_ROI_ROOT = "data/tumor_roi/kits19"`
- `OUT_ROOT = "data/patches/kits19"`
- `PATCH_SIZE = (64, 64, 64)`

Example:

```powershell
kidneycancer extract-patches
```

### `kidneycancer build-2p5d`

Purpose:

- Reads tumor-localized KiTS19 cases
- Builds axial, coronal, and sagittal slice stacks
- Writes outputs to `data/slices_2p5d/kits19`

Config source:

- `src/kidneycancer/slicing_2p5d/config.py`

Important defaults:

- `TUMOR_ROI_ROOT = "data/tumor_roi/kits19"`
- `OUT_ROOT = "data/slices_2p5d/kits19"`
- `SLICE_SIZE = 128`

Example:

```powershell
kidneycancer build-2p5d
```

## 6. Wrapper Scripts

These scripts are thin wrappers around the CLI functionality. They exist mainly for convenience and backward compatibility.

### `scripts/run_preprocessing.py`

Equivalent to:

```powershell
kidneycancer preprocess
```

### `scripts/run_kidney_localization.py`

Equivalent to:

```powershell
kidneycancer localize-kidneys
```

### `scripts/run_tumor_localization.py`

Equivalent to:

```powershell
kidneycancer localize-tumors
```

### `scripts/run_patch_extraction.py`

Equivalent to:

```powershell
kidneycancer extract-patches
```

### `scripts/run_2p5d_slicing.py`

Equivalent to:

```powershell
kidneycancer build-2p5d
```

## 7. Training Scripts

These are experiment-specific scripts. They are not part of the package CLI and they currently use hard-coded dataset paths and output paths.

### `scripts/train_unet3d.py`

Purpose:

- Train a 3D U-Net segmentation baseline on KiTS19 tumor ROI data

Reads from:

- `data/tumor_roi/kits19`

Writes to:

- `checkpoints/unet3d_best.pth`

Run:

```powershell
python scripts/train_unet3d.py
```

### `scripts/train_unetpp.py`

Purpose:

- Train the 3D UNet++ segmentation model

Reads from:

- `data/tumor_roi/kits19`

Writes to:

- `experiments/seg_unetpp/checkpoints/`
- `experiments/seg_unetpp/logs/train_log.csv`

Run:

```powershell
python scripts/train_unetpp.py
```

### `scripts/pretrain_ssl.py`

Purpose:

- Pretrain an EfficientNet encoder using a reconstruction objective on TCGA 2.5D data

Reads from:

- `data/tcga_2p5d`

Writes to:

- `experiments/ssl_b0_encoder.pth`

Run:

```powershell
python scripts/pretrain_ssl.py
```

### `scripts/train_mil.py`

Purpose:

- Train the MIL classifier on TCGA 2.5D slices

Reads from:

- `data/tcga_2p5d`

Optional input:

- `experiments/ssl_b0_encoder.pth`

Writes to:

- `experiments/mil_b0_best.pth`

Run:

```powershell
python scripts/train_mil.py
```

## 8. Inference And Evaluation Scripts

### `scripts/infer_unetpp.py`

Purpose:

- Run sliding-window inference for UNet++
- Save `.npy` predictions and overlay images for a validation subset

Reads from:

- `data/tumor_roi/kits19`
- `experiments/seg_unetpp/checkpoints/best_model.pth`

Writes to:

- `experiments/seg_unetpp/inference_sw`

Run:

```powershell
python scripts/infer_unetpp.py
```

### `scripts/save_qualitative_boundaries.py`

Purpose:

- Save qualitative boundary overlay images and per-case metrics

Reads from:

- `data/tumor_roi/kits19`
- `experiments/seg_unetpp/checkpoints/epoch_010.pth`

Writes to:

- `experiments/seg_unetpp/qualitative/<checkpoint_name>`

Run:

```powershell
python scripts/save_qualitative_boundaries.py
```

### `scripts/infer_tcga_segmentation.py`

Purpose:

- Segment TCGA NIfTI scans using the trained segmentation model

Reads from:

- `data/tcga_nifti`
- `experiments/seg_unetpp/checkpoints/best_model.pth`

Writes to:

- `data/tcga_masks`

Run:

```powershell
python scripts/infer_tcga_segmentation.py
```

### `scripts/infer_tcga_classification.py`

Purpose:

- Run MIL classification on prepared TCGA 2.5D data

Reads from:

- `data/tcga_2p5d`
- `experiments/mil_b0_best.pth`

Writes to:

- `experiments/mil_predictions.csv`
- `experiments/confusion_matrix.png`

Run:

```powershell
python scripts/infer_tcga_classification.py
```

## 9. Data Preparation Utility Scripts

These scripts are utilities for specific preparation steps. They are useful, but they are more specialized than the CLI.

### `scripts/download.py`

Purpose:

- Generate an aria2 download list for KiTS19 imaging volumes
- Launch `aria2c` to download them into `data/raw/kits19`

External dependency:

- `aria2c`

Run:

```powershell
python scripts/download.py
```

### `scripts/copy_segmentation.py`

Purpose:

- Copy KiTS19 segmentation files from a cloned `kits19` repo into `data/raw/kits19`

Assumes source exists under:

- `kits19/data`

Run:

```powershell
python scripts/copy_segmentation.py
```

### `scripts/convert_tcga_dicom_to_nifti.py`

Purpose:

- Convert raw TCGA DICOM studies into gzipped NIfTI files

Reads from:

- `data/raw/tcga_clean`

Writes to:

- `data/tcga_nifti`

External dependency:

- `dcm2niix`

Run:

```powershell
python scripts/convert_tcga_dicom_to_nifti.py
```

### `scripts/lcc_filter.py`

Purpose:

- Keep the largest connected tumor component in predicted TCGA masks

Reads from:

- `data/tcga_masks`

Writes to:

- `data/tcga_masks_lcc`

Run:

```powershell
python scripts/lcc_filter.py
```

### `scripts/extract_roi.py`

Purpose:

- Crop a fixed-size ROI from each TCGA case based on the localized mask

Reads from:

- `data/tcga_nifti`
- `data/tcga_masks_lcc`

Writes to:

- `data/tcga_roi`

Run:

```powershell
python scripts/extract_roi.py
```

### `scripts/extract_2p5d.py`

Purpose:

- Pick the highest-entropy 2.5D slice stack from each TCGA ROI

Reads from:

- `data/tcga_roi`

Writes to:

- `data/tcga_2p5d`

Run:

```powershell
python scripts/extract_2p5d.py
```

### `scripts/check_tcga_shape.py`

Purpose:

- Inspect the set of scan shapes in `data/tcga_nifti`

Run:

```powershell
python scripts/check_tcga_shape.py
```

## 10. Other Scripts

### `scripts/_bootstrap.py`

Purpose:

- Internal helper that adds `src/` to `sys.path` for root-level scripts

Use directly:

- No

### `scripts/summarize_hd95.py`

Status:

- Currently empty

Recommendation:

- Ignore or implement only when you define the exact summary workflow you want

## 11. Recommended End-To-End Workflows

### Workflow A: KiTS19 Data Preparation

```powershell
kidneycancer preprocess
kidneycancer localize-kidneys
kidneycancer localize-tumors
kidneycancer extract-patches
kidneycancer build-2p5d
```

### Workflow B: KiTS19 Segmentation Training

```powershell
python scripts/train_unet3d.py
python scripts/train_unetpp.py
```

### Workflow C: TCGA Segmentation To Classification

```powershell
python scripts/convert_tcga_dicom_to_nifti.py
python scripts/infer_tcga_segmentation.py
python scripts/lcc_filter.py
python scripts/extract_roi.py
python scripts/extract_2p5d.py
python scripts/pretrain_ssl.py
python scripts/train_mil.py
python scripts/infer_tcga_classification.py
```

## 12. Troubleshooting

### `kidneycancer` command not found

Use:

```powershell
python -m kidneycancer --help
```

If that works, your environment is fine and only the shell entry point is missing from `PATH`.

### `ModuleNotFoundError`

Make sure the virtual environment is activated and the project is installed:

```powershell
pip install -e .
```

### `SimpleITK` or `sklearn` import errors

These indicate the active Python environment is missing dependencies. Reinstall in the active environment:

```powershell
pip install -e .
```

### Script runs but uses the wrong folders

Many scripts have hard-coded paths. Review the script or its config module before running it on a different directory layout.

## 13. Recommended User Habits

- Prefer the package CLI for the main KiTS19 preparation steps
- Treat training and inference scripts as experiment entry points
- Review hard-coded paths before running TCGA utility scripts
- Keep all generated data under `data/` and `experiments/` to avoid confusion
