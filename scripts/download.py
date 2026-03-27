import os
import subprocess

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from kidneycancer.utils.logging_utils import configure_logging

SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

LIST_FILE = os.path.join(PROJECT_ROOT, "kits19_targets.txt")


def main():
    logger = configure_logging("kidneycancer.download")
    logger.info("Starting KiTS19 downloader")

    os.chdir(PROJECT_ROOT)
    logger.info("Working directory: %s", os.getcwd())
    logger.info("Generating download list")

    try:
        with open(LIST_FILE, "w", encoding="utf-8") as file_handle:
            for i in range(210):
                case_id = f"{i:05d}"
                rel_dir = os.path.join("data", "raw", "kits19", f"case_{case_id}")
                rel_path = os.path.join(rel_dir, "imaging.nii.gz")
                os.makedirs(rel_dir, exist_ok=True)

                url = (
                    "https://kits19.sfo2.digitaloceanspaces.com/"
                    f"interpolated_{case_id}.nii.gz"
                )
                file_handle.write(f"{url}\n")
                file_handle.write(f"  out={rel_path}\n")

        logger.info("Target list created: %s", LIST_FILE)
    except OSError as exc:
        logger.error("Error creating list file: %s", exc)
        return

    logger.info("Launching aria2c")
    cmd = [
        "aria2c",
        "-i",
        "kits19_targets.txt",
        "-j",
        "8",
        "-x",
        "16",
        "-s",
        "16",
        "-k",
        "1M",
        "--console-log-level=warn",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
    ]

    try:
        subprocess.run(cmd, check=True)
        logger.info("Download complete")

        if os.path.exists(LIST_FILE):
            os.remove(LIST_FILE)
            logger.info("Removed temporary target list")
    except FileNotFoundError:
        logger.error("'aria2c' not found in PATH")
    except subprocess.CalledProcessError as exc:
        logger.error("aria2c failed with exit code %s", exc.returncode)
    except KeyboardInterrupt:
        logger.warning("Stopped by user")


if __name__ == "__main__":
    main()
