"""Inspect the set of NIfTI shapes present in the TCGA input directory."""

from pathlib import Path

import nibabel as nib

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from kidneycancer.utils.logging_utils import configure_logging


ROOT = Path("data/tcga_nifti")


def main() -> None:
    """Print the first file shape and a summary of unique scan shapes."""
    logger = configure_logging("kidneycancer.check_tcga_shape")
    logger.info("Checking first TCGA NIfTI file")
    first_file = next(ROOT.rglob("*.nii.gz"))
    first_image = nib.load(first_file)
    logger.info("First file: %s", first_file.name)
    logger.info("Shape: %s", first_image.shape)

    logger.info("Counting all NIfTI shapes")
    shapes: dict[tuple[int, ...], int] = {}

    for nii_path in ROOT.rglob("*.nii.gz"):
        try:
            image = nib.load(nii_path)
            shape = image.shape
            shapes[shape] = shapes.get(shape, 0) + 1
        except Exception as exc:
            logger.error("Failed to inspect %s: %s", nii_path.name, exc)

    logger.info("Unique shapes summary")
    for shape, count in shapes.items():
        logger.info("%s -> %s scans", shape, count)


if __name__ == "__main__":
    main()
