from pathlib import Path
import sys


def bootstrap_src_path() -> None:
    """Allow repo-root scripts to import the package without installation."""
    src_root = Path(__file__).resolve().parents[1] / "src"
    src_path = str(src_root)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
