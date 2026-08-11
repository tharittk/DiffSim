#!/usr/bin/env python
"""
Slice 3D facies/RMS cubes into paired 2D samples for FlumyDataset.

Reads matched 3D .npy files from facies and RMS directories, extracts
horizontal (z-axis) slices, applies random cropping and augmentation
(rotation, flip), and saves the paired 2D arrays as .npy files for
FlumyDataset to ingest.

The output follows the FlumyDataset convention:
    {output_dir}/{split}/facies/<name>.npy   (H, W) int8
    {output_dir}/{split}/rms/<name>.npy      (H, W) float32

Usage:
    python scripts/generate_2d_slices.py \
        --facies_dir data/flumy3d/facies \
        --rms_dir data/flumy3d/rms \
        --output_dir data/flumy_dataset \
        --crop_size 64 \
        --slices_per_cube 10 \
        --crops_per_slice 4

Everything stays in data/ (legacy dataset3f is untouched).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffsim.data.flumy_generator import FlumyGenerator, map_channelasso_to_litho
from diffsim.data.seismic import compute_facies_mode_window
from diffsim.data.style_match import (
    build_reference_cdf,
    histogram_match_rms,
    load_seismic_rms_grid,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Slice 3D cubes into paired 2D training samples."
    )
    parser.add_argument(
        "--facies_dir",
        type=str,
        default="/mnt/sda_data/tharitt/diffsim/data/flumy3d/facies",
        help="Directory with 3D facies .npy cubes",
    )
    parser.add_argument(
        "--rms_dir",
        type=str,
        default="/mnt/sda_data/tharitt/diffsim/data/flumy3d/rms",
        help="Directory with 3D RMS .npy cubes",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/sda_data/tharitt/diffsim/data/flumy_dataset",
        help="Output root (will contain train/ and test/ subdirs)",
    )
    parser.add_argument(
        "--crop_size", type=int, default=64, help="Spatial crop size (H=W)"
    )
    parser.add_argument(
        "--slices_per_cube",
        type=int,
        default=10,
        help="Number of z-slices to sample per 3D cube",
    )
    parser.add_argument(
        "--crops_per_slice",
        type=int,
        default=16,
        help="Number of random crops per 2D slice",
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.8,
        help="Fraction of samples for training (rest → test)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Apply random rotation and flip augmentation",
    )
    parser.add_argument(
        "--no_augment",
        dest="augment",
        action="store_false",
        help="Disable augmentation",
    )
    parser.add_argument(
        "--rms_window_half",
        type=int,
        default=2,
        help="Half-window size for facies mode computation (must match RMS generation)",
    )
    parser.add_argument(
        "--resample_frames",
        type=int,
        nargs="+",
        default=[256, 128],
        metavar="SIZE",
        help="Frame sizes (H=W) to extract then downsample to crop_size (e.g. 256 128)",
    )
    parser.add_argument(
        "--resamples_per_slice",
        type=int,
        default=4,
        help="Random resample windows to draw per slice per frame size",
    )
    parser.add_argument(
        "--seismic_ref",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to real seismic RMS reference file (e.g. h05_sub1). "
             "When supplied, style matching is applied to every RMS crop.",
    )
    parser.add_argument(
        "--no_style_match",
        action="store_true",
        default=False,
        help="Disable style matching even when --seismic_ref is provided (for debugging).",
    )
    return parser.parse_args()


def random_crop(facies_2d, rms_2d, crop_size, rng):
    """Extract a random crop from aligned facies/RMS 2D arrays."""
    h, w = facies_2d.shape
    if h < crop_size or w < crop_size:
        raise ValueError(
            f"Slice size ({h},{w}) smaller than crop_size ({crop_size}). "
            "Reduce crop_size or increase Flumy grid dimensions."
        )
    y = rng.integers(0, h - crop_size + 1)
    x = rng.integers(0, w - crop_size + 1)
    return (
        facies_2d[y : y + crop_size, x : x + crop_size],
        rms_2d[y : y + crop_size, x : x + crop_size],
    )


def tile_crop(facies_2d, rms_2d, crop_size=64):
    """Split a slice into non-overlapping tiles on a regular grid.

    Input dimensions must be exactly divisible by crop_size.
    Returns a list of (facies, rms) pairs.
    """
    h, w = facies_2d.shape
    if h % crop_size != 0 or w % crop_size != 0:
        raise ValueError(
            f"Input ({h},{w}) not divisible by crop_size ({crop_size})"
        )
    n_rows = h // crop_size
    n_cols = w // crop_size
    tiles = []
    for row in range(n_rows):
        for col in range(n_cols):
            y0 = row * crop_size
            x0 = col * crop_size
            tiles.append(
                (
                    facies_2d[y0 : y0 + crop_size, x0 : x0 + crop_size],
                    rms_2d[y0 : y0 + crop_size, x0 : x0 + crop_size],
                )
            )
    return tiles


def resample_pair(facies_2d, rms_2d, frame_h, frame_w, target_size, rng):
    """Crop a frame_h×frame_w window from the slice and resample to target_size×target_size.

    If frame_h/frame_w equals the slice dimensions, the whole slice is used.
    Facies resampled with nearest-neighbour; RMS with bilinear interpolation.
    """
    from PIL import Image

    h, w = facies_2d.shape
    if frame_h > h or frame_w > w:
        raise ValueError(
            f"frame ({frame_h},{frame_w}) exceeds slice ({h},{w})"
        )

    # Random top-left corner; if frame matches slice, no crop needed
    y = int(rng.integers(0, h - frame_h + 1)) if frame_h < h else 0
    x = int(rng.integers(0, w - frame_w + 1)) if frame_w < w else 0

    fac_window = facies_2d[y : y + frame_h, x : x + frame_w]
    rms_window = rms_2d[y : y + frame_h, x : x + frame_w]

    # Resample
    fac_out = np.array(
        Image.fromarray(fac_window.astype(np.uint8)).resize(
            (target_size, target_size), Image.NEAREST
        ),
        dtype=np.int8,
    )
    rms_out = np.array(
        Image.fromarray(rms_window.astype(np.float32), mode="F").resize(
            (target_size, target_size), Image.BILINEAR
        ),
        dtype=np.float32,
    )
    return fac_out, rms_out


def augment_pair(facies_crop, rms_crop, rng):
    """Apply random rotation (0/90/180/270) and horizontal flip."""
    k = rng.integers(0, 4)
    facies_crop = np.rot90(facies_crop, k=k)
    rms_crop = np.rot90(rms_crop, k=k)

    if rng.random() > 0.5:
        facies_crop = np.fliplr(facies_crop)
        rms_crop = np.fliplr(rms_crop)

    return facies_crop, rms_crop


def main():
    args = parse_args()
    facies_dir = Path(args.facies_dir)
    rms_dir = Path(args.rms_dir)
    output_dir = Path(args.output_dir)
    rng = np.random.default_rng(args.seed)

    # Load real seismic reference for style matching
    ref_sorted_vals = None
    if args.seismic_ref and not args.no_style_match:
        ref_path = Path(args.seismic_ref)
        if not ref_path.is_file():
            print(f"Error: seismic_ref not found: {ref_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading seismic reference: {ref_path}")
        ref_grid = load_seismic_rms_grid(ref_path)
        ref_sorted_vals = build_reference_cdf(ref_grid)
        print(f"  reference grid shape: {ref_grid.shape}, "
              f"value range: {ref_sorted_vals[0]:.2f}–{ref_sorted_vals[-1]:.2f}")
        del ref_grid  # free memory; only CDF is needed

    # Validate inputs
    if not facies_dir.is_dir():
        print(f"Error: {facies_dir} not found.", file=sys.stderr)
        sys.exit(1)
    if not rms_dir.is_dir():
        print(f"Error: {rms_dir} not found.", file=sys.stderr)
        sys.exit(1)

    # Find matched pairs
    facies_files = sorted(facies_dir.glob("*.npy"))
    pairs = []
    for fp in facies_files:
        rp = rms_dir / fp.name
        if rp.is_file():
            pairs.append((fp, rp))
        else:
            print(f"  [warn] No RMS match for {fp.name}, skipping.")

    if not pairs:
        print("Error: no matched facies/RMS cube pairs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pairs)} matched cube pairs")
    print(
        f"  slices_per_cube={args.slices_per_cube}, crops_per_slice={args.crops_per_slice}"
    )
    print(f"  crop_size={args.crop_size}, augment={args.augment}")
    print(f"  resample_frames={args.resample_frames}, resamples_per_slice={args.resamples_per_slice}")
    print()

    # Generate all 2D samples
    samples = []  # list of (facies_2d, rms_2d, name_stem)

    for facies_path, rms_path in pairs:
        print("Processing:", facies_path.name)
        facies_3d = np.load(facies_path)
        rms_3d = np.load(rms_path)

        # RMS has nz-1 depth samples due to reflectivity computation
        nz_facies = facies_3d.shape[2]
        nz_rms = rms_3d.shape[2]

        # Select z-slices (sample from the RMS-valid range)
        nz = min(nz_facies, nz_rms)
        # Evenly spaced for random crops
        z_indices_rand = np.linspace(0, nz - 1, num=args.slices_per_cube, dtype=int)
        # Random selection for tile crops (different z-levels → 2x coverage)
        z_indices_tile = rng.choice(
            nz, size=min(args.slices_per_cube, nz), replace=False
        )
        z_indices_tile.sort()

        stem = facies_path.stem  # e.g. "ng50_isbx80_seed0"

        facies_3d = FlumyGenerator.reclassify_facies(facies_3d)

        # Compute windowed facies mode (consistent with RMS averaging window)
        facies_mode_3d = compute_facies_mode_window(facies_3d, args.rms_window_half)
        # Final training target is LITHO_FLUMY_AV (0=shale,1=sand,2=silt)
        facies_mode_3d = map_channelasso_to_litho(facies_mode_3d)

        for z in z_indices_rand:
            facies_slice = facies_mode_3d[:, :, z]
            rms_slice = rms_3d[:, :, z]

            for crop_idx in range(args.crops_per_slice):
                facies_crop, rms_crop = random_crop(
                    facies_slice, rms_slice, args.crop_size, rng
                )

                if args.augment:
                    facies_crop, rms_crop = augment_pair(facies_crop, rms_crop, rng)

                # Ensure contiguous arrays with correct dtypes
                facies_crop = np.ascontiguousarray(facies_crop, dtype=np.int8)
                rms_crop = np.ascontiguousarray(rms_crop, dtype=np.float32)

                name = f"{stem}_z{z:03d}_c{crop_idx}_rand"
                samples.append((facies_crop, rms_crop, name))

        # Tile-based cropping (no randomness, just a regular grid)
        for z in z_indices_tile:
            facies_slice = facies_mode_3d[:, :, z]
            rms_slice = rms_3d[:, :, z]

            tiles = tile_crop(facies_slice, rms_slice, crop_size=args.crop_size)

            for crop_idx, (facies_crop, rms_crop) in enumerate(tiles):
                if args.augment:
                    facies_crop, rms_crop = augment_pair(facies_crop, rms_crop, rng)

                # Ensure contiguous arrays with correct dtypes
                facies_crop = np.ascontiguousarray(facies_crop, dtype=np.int8)
                rms_crop = np.ascontiguousarray(rms_crop, dtype=np.float32)

                name = f"{stem}_z{z:03d}_t{crop_idx}_tile"
                samples.append((facies_crop, rms_crop, name))

        # Resample: extract arbitrary-sized frame and downsample to crop_size
        z_indices_resamp = rng.choice(
            nz, size=min(args.slices_per_cube, nz), replace=False
        )
        for z in z_indices_resamp:
            facies_slice = facies_mode_3d[:, :, z]
            rms_slice = rms_3d[:, :, z]
            h_sl, w_sl = facies_slice.shape

            for frame_size in args.resample_frames:
                if frame_size > h_sl or frame_size > w_sl:
                    continue  # frame larger than slice; skip silently

                for r_idx in range(args.resamples_per_slice):
                    facies_crop, rms_crop = resample_pair(
                        facies_slice, rms_slice,
                        frame_size, frame_size,
                        args.crop_size, rng,
                    )

                    if args.augment:
                        facies_crop, rms_crop = augment_pair(facies_crop, rms_crop, rng)

                    name = f"{stem}_z{z:03d}_f{frame_size}_r{r_idx}_resamp"
                    samples.append((facies_crop, rms_crop, name))

    slice_h, slice_w = facies_mode_3d.shape[:2]  # representative slice shape
    n_tiles = (slice_h // args.crop_size) * (slice_w // args.crop_size)
    valid_frames = [f for f in args.resample_frames if f <= slice_h and f <= slice_w]
    print(
        f"  per cube: {len(z_indices_rand) * args.crops_per_slice} random crops, "
        f"{len(z_indices_tile) * n_tiles} tiles, "
        f"{len(z_indices_resamp) * len(valid_frames) * args.resamples_per_slice} resampled"
    )

    print(f"Generated {len(samples)} total 2D samples")

    # Shuffle and split into train/test
    rng.shuffle(samples)
    n_train = int(len(samples) * args.train_fraction)
    train_samples = samples[:n_train]
    test_samples = samples[n_train:]

    print(f"  train: {len(train_samples)}, test: {len(test_samples)}")

    # Save
    for split_name, split_samples in [("train", train_samples), ("test", test_samples)]:
        facies_out = output_dir / split_name / "facies"
        rms_out = output_dir / split_name / "rms"
        facies_out.mkdir(parents=True, exist_ok=True)
        rms_out.mkdir(parents=True, exist_ok=True)

        for facies_crop, rms_crop, name in split_samples:
            if ref_sorted_vals is not None:
                rms_crop = histogram_match_rms(rms_crop, ref_sorted_vals)
            np.save(facies_out / f"{name}.npy", facies_crop)
            np.save(rms_out / f"{name}.npy", rms_crop)

    if ref_sorted_vals is not None:
        print("  style matching applied (histogram transfer from seismic reference)")
    print(f"\nSaved to {output_dir}/{{train,test}}/{{facies,rms}}/")
    print("Done.")


if __name__ == "__main__":
    main()
