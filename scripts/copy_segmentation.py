import os
import shutil

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from kidneycancer.utils.logging_utils import configure_logging

SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

SOURCE_BASE = os.path.join(PROJECT_ROOT, "kits19", "data")
DEST_BASE = os.path.join(PROJECT_ROOT, "data", "raw", "kits19")


def main():
    logger = configure_logging("kidneycancer.copy_segmentation")
    logger.info("Copying KiTS19 segmentations")
    logger.info("Source: %s", SOURCE_BASE)
    logger.info("Destination: %s", DEST_BASE)

    if not os.path.exists(SOURCE_BASE):
        logger.error("Source directory not found: %s", SOURCE_BASE)
        logger.error("Clone the kits19 repository first if the source is missing")
        return

    count = 0
    missing = 0

    for i in range(210):
        case_id = f"case_{i:05d}"
        src_file = os.path.join(SOURCE_BASE, case_id, "segmentation.nii.gz")
        dest_dir = os.path.join(DEST_BASE, case_id)
        dest_file = os.path.join(dest_dir, "segmentation.nii.gz")

        if os.path.exists(src_file):
            os.makedirs(dest_dir, exist_ok=True)
            try:
                shutil.copy2(src_file, dest_file)
                logger.info("Copied %s", case_id)
                count += 1
            except OSError as exc:
                logger.error("Failed to copy %s: %s", case_id, exc)
        else:
            logger.warning("Missing segmentation in source: %s", case_id)
            missing += 1

    logger.info("Transferred %s segmentation files", count)
    if missing > 0:
        logger.warning(
            "%s cases did not have segmentations in the source directory",
            missing,
        )


if __name__ == "__main__":
    main()
