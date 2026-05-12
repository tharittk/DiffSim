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

from diffsim.data.flumy_generator import FlumyGenerator


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
    # output local for fast(er) training
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/flumy_dataset",
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
        default=4,
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
    """Split a 256x256 slice into non-overlapping tiles on a regular grid.

    Assumes input is exactly 256x256. With crop_size=64 this yields a 4x4
    grid of 16 tiles. Returns a list of (facies, rms) pairs.
    """
    h, w = facies_2d.shape
    if h != 256 or w != 256:
        raise ValueError(f"Expected 256x256 input, got ({h},{w})")
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
    print()

    # Generate all 2D samples
    samples = []  # list of (facies_2d, rms_2d, name_stem)

    for facies_path, rms_path in pairs:
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

        facies_3d = FlumyGenerator.reclassify_to_three_facies(facies_3d)

        for z in z_indices_rand:
            facies_slice = facies_3d[:, :, z]
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
            facies_slice = facies_3d[:, :, z]
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

    print(
        f"generated {len(z_indices_rand) * args.crops_per_slice} random crops per cube"
    )
    print(f"generated {len(z_indices_tile) * len(tiles)} tile crops per cube")

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
            np.save(facies_out / f"{name}.npy", facies_crop)
            np.save(rms_out / f"{name}.npy", rms_crop)

    print(f"\nSaved to {output_dir}/{{train,test}}/{{facies,rms}}/")
    print("Done.")


if __name__ == "__main__":
    main()
