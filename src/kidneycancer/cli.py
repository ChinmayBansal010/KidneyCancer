"""Command-line interface for the kidneycancer package."""

from __future__ import annotations

import argparse
from pathlib import Path

from kidneycancer.utils.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kidneycancer",
        description="Kidney cancer data preparation pipeline utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess the raw KiTS19 dataset.")
    preprocess_parser.add_argument("--input-root", type=Path, default=None, help="Override the raw KiTS19 input root.")
    preprocess_parser.add_argument("--output-root", type=Path, default=None, help="Override the preprocessed output root.")
    preprocess_parser.add_argument("--workers", type=int, default=None, help="Number of worker processes to use.")

    kidney_parser = subparsers.add_parser(
        "localize-kidneys",
        help="Crop preprocessed volumes down to kidney regions.",
    )
    kidney_parser.add_argument("--input-root", type=Path, default=None, help="Override the preprocessed input root.")
    kidney_parser.add_argument("--output-root", type=Path, default=None, help="Override the kidney ROI output root.")

    tumor_parser = subparsers.add_parser(
        "localize-tumors",
        help="Crop kidney volumes down to tumor regions when present.",
    )
    tumor_parser.add_argument("--input-root", type=Path, default=None, help="Override the kidney ROI input root.")
    tumor_parser.add_argument("--output-root", type=Path, default=None, help="Override the tumor ROI output root.")

    patch_parser = subparsers.add_parser(
        "extract-patches",
        help="Extract training patches from tumor-localized volumes.",
    )
    patch_parser.add_argument("--input-root", type=Path, default=None, help="Override the tumor ROI input root.")
    patch_parser.add_argument("--output-root", type=Path, default=None, help="Override the patch output root.")

    slice_parser = subparsers.add_parser(
        "build-2p5d",
        help="Build 2.5D slices from tumor-localized volumes.",
    )
    slice_parser.add_argument("--input-root", type=Path, default=None, help="Override the tumor ROI input root.")
    slice_parser.add_argument("--output-root", type=Path, default=None, help="Override the 2.5D slice output root.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = configure_logging("kidneycancer.cli")

    commands = {
        "preprocess": _run_preprocess,
        "localize-kidneys": _run_localize_kidneys,
        "localize-tumors": _run_localize_tumors,
        "extract-patches": _run_extract_patches,
        "build-2p5d": _run_build_slices,
    }
    try:
        commands[args.command](args)
    except Exception:
        logger.exception("Command '%s' failed", args.command)
        return 1
    return 0


def _run_preprocess(args: argparse.Namespace) -> None:
    from kidneycancer.preprocessing.preprocess_dataset import preprocess_all

    preprocess_all(
        raw_root=args.input_root if args.input_root else None,
        out_root=args.output_root if args.output_root else None,
        workers=args.workers,
    )


def _run_localize_kidneys(args: argparse.Namespace) -> None:
    from kidneycancer.kidney_localization.localize_dataset import localize_all

    localize_all(
        preprocessed_root=args.input_root if args.input_root else None,
        out_root=args.output_root if args.output_root else None,
    )


def _run_localize_tumors(args: argparse.Namespace) -> None:
    from kidneycancer.tumor_localization.localize_dataset import localize_all

    localize_all(
        kidney_roi_root=args.input_root if args.input_root else None,
        out_root=args.output_root if args.output_root else None,
    )


def _run_extract_patches(args: argparse.Namespace) -> None:
    from kidneycancer.patch_extraction.extract_patches import extract_all

    extract_all(
        tumor_roi_root=args.input_root if args.input_root else None,
        out_root=args.output_root if args.output_root else None,
    )


def _run_build_slices(args: argparse.Namespace) -> None:
    from kidneycancer.slicing_2p5d.build_slices import build_all

    build_all(
        tumor_roi_root=args.input_root if args.input_root else None,
        out_root=args.output_root if args.output_root else None,
    )
