#!/usr/bin/env python3
"""
slice_kits19.py
Slice KiTS19 preprocessed NIfTIs into 2D PNGs (512x512), generate masks, bbox JSON, and per-case metadata.

Inputs (per case directory):
 - imaging_preprocessed.nii.gz   (float32, normalized or resampled)
 - segmentation_preprocessed.nii.gz  (uint8 or float; labels: 0 background, 1 kidney, 2 tumor)

Outputs:
 slices/kits19/case_xxxxx/
   img_000.png
   mask_000.png
   bbox_000.json
   overlay_000.png  (optional)
   meta.json

Usage:
 python scripts/slice_kits19.py -i data/processed/kits19_preprocessed -o data/slices/kits19 --workers 2
"""

from pathlib import Path
import argparse
import json
import os
from multiprocessing import Pool, cpu_count, current_process
from functools import partial

import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw
from tqdm import tqdm

# ------------- Helpers -------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_png(arr_uint8: np.ndarray, out_path: Path):
    img = Image.fromarray(arr_uint8)
    img.save(str(out_path), format="PNG", compress_level=1)  # low compression = faster

def compute_bbox_from_mask(mask2d):
    """Return bbox [xmin, ymin, xmax, ymax] or None if empty."""
    coords = np.argwhere(mask2d > 0)
    if coords.size == 0:
        return None
    ys, xs = coords[:,0], coords[:,1]
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())
    return [int(xmin), int(ymin), int(xmax), int(ymax)]

def make_overlay(img_u8, mask_u8, alpha=0.5):
    """Simple RGB overlay: red for tumor (2), green for kidney (1)."""
    img = Image.fromarray(img_u8).convert("RGB")
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    mask_arr = np.array(mask_u8)
    # kidney (label==1) -> green
    kidney = (mask_arr == 1).astype(np.uint8) * 255
    tumor = (mask_arr == 2).astype(np.uint8) * 255
    # draw pixels as filled rectangles on overlay for performance use putdata
    overlay_pixels = np.array(overlay)
    # create color layers
    overlay_pixels[...,0] = np.where(tumor==255, 255, overlay_pixels[...,0])  # red for tumor
    overlay_pixels[...,1] = np.where(kidney==255, 180, overlay_pixels[...,1])  # green-ish for kidney
    overlay_pixels[...,3] = np.where((tumor==255)|(kidney==255), int(255*alpha), overlay_pixels[...,3])
    overlay = Image.fromarray(overlay_pixels, "RGBA")
    out = Image.alpha_composite(img.convert("RGBA"), overlay)
    return out.convert("RGB")

# ------------- Scaling -------------
def scale_zclip_to_uint8(volume, clip_z=5.0):
    """
    volume: float (z-scored)
    clip_z: half-window in z-units to map to 0..255
    maps -clip_z -> 0 and +clip_z -> 255 (linear)
    """
    v = np.clip(volume, -clip_z, clip_z)
    v = (v + clip_z) / (2 * clip_z)  # 0..1
    v = (v * 255.0).astype(np.uint8)
    return v

def scale_percentile_to_uint8(volume, pmin=0.5, pmax=99.5):
    """Compute volume-level percentile scaling to 0..255."""
    lo = np.percentile(volume, pmin)
    hi = np.percentile(volume, pmax)
    if hi <= lo:
        hi = lo + 1e-3
    v = np.clip((volume - lo) / (hi - lo), 0.0, 1.0)
    v = (v * 255.0).astype(np.uint8)
    return v

# ------------- Per-case processing -------------
def process_case(case_dir: Path, out_root: Path,
                 resize_hw=(512,512),
                 skip_empty=True,
                 min_tumor_px=1,
                 scale_mode="zclip",
                 clip_z=5.0,
                 pmin=0.5, pmax=99.5,
                 write_overlay=True):
    pid = current_process().pid
    case_name = case_dir.name
    out_case = out_root / case_name
    ensure_dir(out_case)

    img_nii_path = case_dir / "imaging_preprocessed.nii.gz"
    seg_nii_path = case_dir / "segmentation_preprocessed.nii.gz"
    meta = {
        "case_id": case_name,
        "slices": [],
    }

    if not img_nii_path.exists():
        return f"[SKIP] {case_name} missing imaging_preprocessed.nii.gz"
    if not seg_nii_path.exists():
        return f"[SKIP] {case_name} missing segmentation_preprocessed.nii.gz"

    try:
        img_nii = nib.load(str(img_nii_path))
        seg_nii = nib.load(str(seg_nii_path))
    except Exception as e:
        return f"[ERR] {case_name} nib load failed: {e}"

    img_vol = img_nii.get_fdata(dtype=np.float32)  # shape Z,H,W
    seg_vol = seg_nii.get_fdata(dtype=np.float32)

    if img_vol.ndim != 3:
        return f"[ERR] {case_name} imaging not 3D"

    Z = min(img_vol.shape[0], seg_vol.shape[0])
    H, W = img_vol.shape[1], img_vol.shape[2]

    # choose scaling mode
    if scale_mode == "zclip":
        # scale whole volume to uint8 using z clipping
        img_u8_vol = scale_zclip_to_uint8(img_vol, clip_z=clip_z)
    else:
        img_u8_vol = scale_percentile_to_uint8(img_vol, pmin=pmin, pmax=pmax)

    # iterate slices
    saved_count = 0
    for z in range(Z):
        img2d_u8 = img_u8_vol[z]  # uint8 H x W
        mask2d = seg_vol[z].astype(np.uint8)  # values expected 0/1/2

        # optionally skip empty slices
        if skip_empty and mask2d.sum() == 0:
            continue

        # check tumor pixels
        tumor_px = int((mask2d == 2).sum())
        has_tumor = tumor_px >= min_tumor_px

        if skip_empty and (not has_tumor) and (mask2d.sum() == 0):
            continue

        # Resize to 512x512
        # Image: bilinear, Mask: nearest
        img_pil = Image.fromarray(img2d_u8)
        img_resized = img_pil.resize(resize_hw, resample=Image.BILINEAR)
        mask_pil = Image.fromarray(mask2d)
        mask_resized = mask_pil.resize(resize_hw, resample=Image.NEAREST)

        idx = saved_count
        img_name = f"img_{idx:04d}.png"
        mask_name = f"mask_{idx:04d}.png"
        bbox_name = f"bbox_{idx:04d}.json"

        # compute bbox on resized mask
        mask_arr_resized = np.array(mask_resized)
        kidney_bbox = compute_bbox_from_mask((mask_arr_resized == 1).astype(np.uint8))
        tumor_bbox = compute_bbox_from_mask((mask_arr_resized == 2).astype(np.uint8))
        combined_bbox = compute_bbox_from_mask((mask_arr_resized > 0).astype(np.uint8))

        # Save files
        save_png(np.array(img_resized), out_case / img_name)
        save_png(np.array(mask_resized).astype(np.uint8), out_case / mask_name)

        bbox_obj = {
            "slice_index_orig": int(z),
            "img_filename": img_name,
            "mask_filename": mask_name,
            "kidney_bbox": kidney_bbox,
            "tumor_bbox": tumor_bbox,
            "combined_bbox": combined_bbox,
            "has_tumor": bool(has_tumor),
            "tumor_pixels_original": int(tumor_px)
        }
        with open(out_case / bbox_name, "w") as f:
            json.dump(bbox_obj, f)

        if write_overlay:
            try:
                overlay = make_overlay(np.array(img_resized), np.array(mask_resized), alpha=0.45)
                overlay.save(str(out_case / f"overlay_{idx:04d}.png"), format="PNG")
            except Exception:
                pass

        meta["slices"].append({
            "slice_id": idx,
            "orig_index": int(z),
            "img": img_name,
            "mask": mask_name,
            "bbox": bbox_name,
            "has_tumor": bool(has_tumor),
            "tumor_px": int(tumor_px)
        })

        saved_count += 1

    # write meta.json
    meta["saved_slices"] = saved_count
    meta["orig_shape"] = [int(Z), int(H), int(W)]
    meta["resize"] = list(resize_hw)
    meta["scale_mode"] = scale_mode
    meta["clip_z"] = clip_z
    meta["percentile_pmin_pmax"] = [pmin, pmax]

    with open(out_case / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return f"[OK] {case_name} saved={saved_count}"

# -------------- CLI / Multiprocessing --------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="input preprocessed kits19 root (folders case_xxx)")
    p.add_argument("--output", "-o", required=True, help="output slices root (will create kits19/CASE/ files)")
    p.add_argument("--workers", "-w", type=int, default=None)
    p.add_argument("--resize", type=int, nargs=2, default=[512,512], help="HxW target size")
    p.add_argument("--skip-empty", action="store_true", help="skip slices with empty mask")
    p.add_argument("--min-tumor-px", type=int, default=1, help="minimum tumor pixels to consider has_tumor")
    p.add_argument("--scale-mode", choices=["zclip","percentile"], default="zclip")
    p.add_argument("--clip-z", type=float, default=5.0, help="z clip half-window when using zclip")
    p.add_argument("--pmin", type=float, default=0.5, help="percentile min")
    p.add_argument("--pmax", type=float, default=99.5, help="percentile max")
    p.add_argument("--no-overlay", dest="overlay", action="store_false")
    return p.parse_args()

def worker_entry(args):
    case_dir, out_root, cfg = args
    return process_case(case_dir, out_root, **cfg)

def main():
    args = parse_args()
    IN = Path(args.input)
    OUT = Path(args.output)
    ensure_dir(OUT)

    cases = sorted([d for d in IN.iterdir() if d.is_dir() and d.name.startswith("case_")])
    if not cases:
        print("[ERR] no cases found"); return

    # workers safe default
    cpu_cnt = cpu_count()
    workers = args.workers if args.workers and args.workers > 0 else min(4, cpu_cnt)
    workers = max(1, workers)

    cfg = {
        "resize_hw": tuple(args.resize),
        "skip_empty": args.skip_empty,
        "min_tumor_px": args.min_tumor_px,
        "scale_mode": args.scale_mode,
        "clip_z": args.clip_z,
        "pmin": args.pmin,
        "pmax": args.pmax,
        "write_overlay": args.overlay
    }

    tasks = [(d, OUT, cfg) for d in cases]

    print(f"Found {len(cases)} cases. Workers: {workers}. Output: {OUT}")
    results = []
    if workers > 1:
        with Pool(workers) as pool:
            for r in tqdm(pool.imap_unordered(worker_entry, tasks), total=len(tasks)):
                print(r)
                results.append(r)
    else:
        for t in tqdm(tasks):
            r = worker_entry(t)
            print(r)
            results.append(r)

    print("Done. Summary:")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
