#!/usr/bin/env python
"""
Convert 3D facies .npy cubes to 3D RMS amplitude cubes via seismic forward modeling.

Reads all .npy files from the facies input directory, runs the seismic pipeline
(facies → AI → reflectivity → synthetic seismic → RMS), and saves the output
RMS cubes to the output directory with the same filenames.

Usage:
    python scripts/generate_rms_cubes.py \
        --input_dir data/flumy3d/facies \
        --output_dir data/flumy3d/rms

No shell script needed — this processes all files in the input directory.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffsim.data.seismic import generate_rms_from_facies_3d
from diffsim.data.flumy_generator import FlumyGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert 3D facies cubes to RMS amplitude cubes."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/flumy3d/facies",
        help="Directory with 3D facies .npy files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/flumy3d/rms",
        help="Output directory for RMS .npy files",
    )
    # parser.add_argument(
    #     "--f_dominant",
    #     type=float,
    #     default=1000.0,
    #     help="Dominant frequency of Ricker wavelet (Hz)",
    # )
    # the nz of generated cube is 30. 14 is about half.
    parser.add_argument(
        "--rms_window_half",
        type=int,
        default=1,
        help="Half-window size for RMS computation (samples)",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.2,
        help="Additive noise level (relative to signal std)",
    )
    # parser.add_argument(
    #     "--smooth_sigma",
    #     type=float,
    #     default=1.0,
    #     help="Lateral Gaussian smoothing sigma (grid cells)",
    # )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for noise and stochastic rock properties",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(f"Error: input directory {input_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    npy_files = sorted(input_dir.glob("*.npy"))
    if not npy_files:
        print(f"Error: no .npy files found in {input_dir}.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(npy_files)} facies cubes → RMS cubes")
    # print(f"  f_dominant={args.f_dominant}, rms_window_half={args.rms_window_half}")
    # print(f"  noise_level={args.noise_level}, smooth_sigma={args.smooth_sigma}")
    # print()

    rng = np.random.default_rng(args.seed)

    for i, facies_path in enumerate(npy_files, 1):
        out_path = output_dir / facies_path.name

        if out_path.exists():
            print(f"  [{i}/{len(npy_files)}] [skip] {out_path.name} already exists")
            continue

        t0 = time.time()
        facies_block = np.load(facies_path)
        facies_block = FlumyGenerator.reclassify_to_three_facies(facies_block)

        rms_cube = generate_rms_from_facies_3d(
            facies_block,
            rms_window_half=args.rms_window_half,
            noise_level=args.noise_level,
            smooth_sigma=None,  # auto-compute from Fresnel zone radius
            rng=rng,
        )
        np.save(out_path, rms_cube)
        elapsed = time.time() - t0
        print(
            f"  [{i}/{len(npy_files)}] {facies_path.name}  "
            f"facies={facies_block.shape} → rms={rms_cube.shape}  {elapsed:.1f}s"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
