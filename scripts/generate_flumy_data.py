#!/usr/bin/env python
"""
Generate training data from Flumy simulations.

This script runs multiple Flumy simulations with varied geological parameters,
extracts 2D plan-view facies maps, computes RMS amplitude attributes via
seismic forward modeling, and saves paired (facies, RMS) data as .npz files
for training the conditional diffusion model.

The seismic forward modeling pipeline:
    1. Facies → Acoustic Impedance (per facies type)
    2. AI → Vertical reflectivity coefficients
    3. Reflectivity → Synthetic seismic (Ricker wavelet convolution)
    4. Synthetic seismic → RMS amplitude (depth-windowed)

Usage:
    python scripts/generate_flumy_data.py \\
        --output_dir data/flumy_dataset \\
        --n_simulations 100 \\
        --slices_per_sim 10 \\
        --image_size 64

    # Or from a config file:
    python scripts/generate_flumy_data.py --config configs/case1_flumy.json
"""

import argparse
import json
import os
import sys

import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffsim.data.flumy_generator import FlumyGenerator
from diffsim.data.seismic import (
    generate_rms_from_facies_3d,
    generate_rms_from_facies_2d,
)


def _resize_nearest(arr, target_size):
    """Resize with nearest neighbor (for discrete facies)."""
    from PIL import Image

    img = Image.fromarray(arr.astype(np.uint8))
    img = img.resize((target_size[1], target_size[0]), Image.NEAREST)
    return np.array(img).astype(np.int8)


def _resize_bilinear(arr, target_size):
    """Resize with bilinear interpolation (for continuous values)."""
    from PIL import Image

    img = Image.fromarray(arr.astype(np.float32), mode="F")
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    return np.array(img, dtype=np.float32)


def generate_dataset(
    output_dir,
    n_simulations=100,
    slices_per_sim=10,
    image_size=64,
    seed=42,
    # Flumy parameters
    nx=250,
    ny=250,
    mesh=10,
    hmax=3.0,
    ng_range=(30, 70),
    isbx_range=(50, 100),
    # Seismic parameters
    f_dominant=25.0,
    rms_window_half=5,
    noise_level=0.05,
    smooth_sigma=1.0,
    # Split
    train_fraction=0.8,
):
    """
    Generate a complete training dataset from Flumy simulations.

    For each simulation:
        1. Run Flumy with randomized geological parameters
        2. Extract plan-view slices at multiple depth levels
        3. For each slice, compute RMS amplitude from the full 3D block
        4. Save (facies, RMS) pairs as compressed .npz files

    Args:
        output_dir: Base output directory (train/ and test/ subdirs created)
        n_simulations: Number of Flumy simulations to run
        slices_per_sim: Max number of plan-view slices per simulation
        image_size: Target image size (square)
        seed: Base random seed
        nx, ny: Flumy grid dimensions (number of nodes)
        mesh: Flumy grid cell size in meters
        hmax: Maximum channel depth in meters
        ng_range: (min, max) net-to-gross ratio to sample from
        isbx_range: (min, max) sand body extension parameter
        f_dominant: Ricker wavelet dominant frequency in Hz
        rms_window_half: Half-window size for RMS computation (samples)
        noise_level: Additive noise level for RMS (relative to signal std)
        smooth_sigma: Lateral Gaussian smoothing for RMS (grid cells)
        train_fraction: Fraction of simulations for training set
    """
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    n_train = int(n_simulations * train_fraction)
    sample_idx = 0

    print(f"=== Flumy Training Data Generation ===")
    print(
        f"Simulations: {n_simulations} ({n_train} train, "
        f"{n_simulations - n_train} test)"
    )
    print(f"Slices per sim: {slices_per_sim}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Output: {output_dir}")
    print(f"Flumy grid: {nx}x{ny}, mesh={mesh}m, hmax={hmax}m")
    print(
        f"Seismic: f_dom={f_dominant}Hz, RMS window={2*rms_window_half+1}, "
        f"noise={noise_level}, smooth={smooth_sigma}"
    )
    print()

    for sim_idx in tqdm(range(n_simulations), desc="Simulations"):
        # Vary geological parameters for diversity
        ng = int(rng.integers(ng_range[0], ng_range[1] + 1))
        isbx = int(rng.integers(isbx_range[0], isbx_range[1] + 1))
        sim_seed = int(rng.integers(0, 2**31))

        generator = FlumyGenerator(
            nx=nx,
            ny=ny,
            mesh=mesh,
            hmax=hmax,
            ng=ng,
            isbx=isbx,
            verbose=False,
        )

        try:
            facies_block = generator.generate(sim_seed)
        except Exception as e:
            print(f"\n  Simulation {sim_idx} failed (seed={sim_seed}): {e}")
            continue

        # Select depth indices for plan-view extraction
        nz = facies_block.shape[2]
        if nz < 3:
            print(f"\n  Simulation {sim_idx}: too few depth layers ({nz}), skipping")
            continue

        # Skip top/bottom boundaries, sample from interior
        z_margin = max(2, nz // 10)
        available_z = list(range(z_margin, nz - z_margin))
        if len(available_z) == 0:
            available_z = list(range(nz))

        n_slices = min(slices_per_sim, len(available_z))
        z_indices = rng.choice(available_z, size=n_slices, replace=False)

        for z_idx in z_indices:
            # Extract plan-view facies
            facies_2d = facies_block[:, :, int(z_idx)]

            # Skip slices that are all one facies (not interesting for training)
            if len(np.unique(facies_2d)) < 2:
                continue

            # Compute RMS from 3D block (physically rigorous)
            rms_2d = generate_rms_from_facies_3d(
                facies_block,
                z_target=int(z_idx),
                f_dominant=f_dominant,
                rms_window_half=rms_window_half,
                noise_level=noise_level,
                smooth_sigma=smooth_sigma,
                rng=rng,
            )

            # Resize to target image size
            target = (image_size, image_size)
            facies_resized = _resize_nearest(facies_2d, target)
            rms_resized = _resize_bilinear(rms_2d, target)

            # Determine train vs test split (by simulation, not by slice)
            save_dir = train_dir if sim_idx < n_train else test_dir

            # Save as compressed .npz
            fname = f"sample_{sample_idx:06d}.npz"
            np.savez_compressed(
                save_dir / fname,
                facies=facies_resized.astype(np.int8),
                rms=rms_resized.astype(np.float32),
            )
            sample_idx += 1

    print(f"\n=== Generation Complete ===")
    n_train_files = len(list(train_dir.glob("*.npz")))
    n_test_files = len(list(test_dir.glob("*.npz")))
    print(f"Total samples: {sample_idx}")
    print(f"  Train: {n_train_files} files in {train_dir}")
    print(f"  Test:  {n_test_files} files in {test_dir}")

    # Save generation metadata for reproducibility
    metadata = {
        "n_simulations": n_simulations,
        "slices_per_sim": slices_per_sim,
        "image_size": image_size,
        "seed": seed,
        "nx": nx,
        "ny": ny,
        "mesh": mesh,
        "hmax": hmax,
        "ng_range": list(ng_range),
        "isbx_range": list(isbx_range),
        "f_dominant": f_dominant,
        "rms_window_half": rms_window_half,
        "noise_level": noise_level,
        "smooth_sigma": smooth_sigma,
        "train_fraction": train_fraction,
        "n_train_files": n_train_files,
        "n_test_files": n_test_files,
        "total_samples": sample_idx,
    }
    metadata_path = output_dir / "generation_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data from Flumy simulations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file option
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to config JSON file (overrides other arguments)",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/flumy_dataset",
        help="Output directory for generated data",
    )

    # Dataset size
    parser.add_argument(
        "--n_simulations",
        type=int,
        default=100,
        help="Number of Flumy simulations to run",
    )
    parser.add_argument(
        "--slices_per_sim",
        type=int,
        default=10,
        help="Max plan-view slices per simulation",
    )
    parser.add_argument(
        "--image_size", type=int, default=64, help="Output image size (square)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Flumy parameters
    parser.add_argument("--nx", type=int, default=250, help="Grid nodes along x")
    parser.add_argument("--ny", type=int, default=250, help="Grid nodes along y")
    parser.add_argument("--mesh", type=int, default=10, help="Grid mesh size (m)")
    parser.add_argument("--hmax", type=float, default=3.0, help="Max channel depth (m)")
    parser.add_argument("--ng_min", type=int, default=30, help="Min net-to-gross (%%)")
    parser.add_argument("--ng_max", type=int, default=70, help="Max net-to-gross (%%)")
    parser.add_argument(
        "--isbx_min", type=int, default=50, help="Min sand body extension"
    )
    parser.add_argument(
        "--isbx_max", type=int, default=100, help="Max sand body extension"
    )

    # Seismic parameters
    parser.add_argument(
        "--f_dominant", type=float, default=25.0, help="Ricker wavelet frequency (Hz)"
    )
    parser.add_argument(
        "--rms_window_half", type=int, default=5, help="RMS window half-size (samples)"
    )
    parser.add_argument(
        "--noise_level", type=float, default=0.05, help="RMS noise level (relative)"
    )
    parser.add_argument(
        "--smooth_sigma",
        type=float,
        default=1.0,
        help="RMS lateral smoothing sigma (grid cells)",
    )

    # Split
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.8,
        help="Fraction of simulations for training",
    )

    args = parser.parse_args()

    # If config file provided, load parameters from it
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        dg = config.get("data_generation", {})
        flumy_cfg = dg.get("flumy", {})
        seismic_cfg = dg.get("seismic", {})

        generate_dataset(
            output_dir=dg.get("output_dir", args.output_dir),
            n_simulations=dg.get("n_simulations", args.n_simulations),
            slices_per_sim=dg.get("slices_per_sim", args.slices_per_sim),
            image_size=config.get("image_size", args.image_size),
            seed=dg.get("seed", args.seed),
            nx=flumy_cfg.get("nx", args.nx),
            ny=flumy_cfg.get("ny", args.ny),
            mesh=flumy_cfg.get("mesh", args.mesh),
            hmax=flumy_cfg.get("hmax", args.hmax),
            ng_range=tuple(flumy_cfg.get("ng_range", [args.ng_min, args.ng_max])),
            isbx_range=tuple(
                flumy_cfg.get("isbx_range", [args.isbx_min, args.isbx_max])
            ),
            f_dominant=seismic_cfg.get("f_dominant", args.f_dominant),
            rms_window_half=seismic_cfg.get("rms_window_half", args.rms_window_half),
            noise_level=seismic_cfg.get("noise_level", args.noise_level),
            smooth_sigma=seismic_cfg.get("smooth_sigma", args.smooth_sigma),
            train_fraction=dg.get("train_fraction", args.train_fraction),
        )
    else:
        generate_dataset(
            output_dir=args.output_dir,
            n_simulations=args.n_simulations,
            slices_per_sim=args.slices_per_sim,
            image_size=args.image_size,
            seed=args.seed,
            nx=args.nx,
            ny=args.ny,
            mesh=args.mesh,
            hmax=args.hmax,
            ng_range=(args.ng_min, args.ng_max),
            isbx_range=(args.isbx_min, args.isbx_max),
            f_dominant=args.f_dominant,
            rms_window_half=args.rms_window_half,
            noise_level=args.noise_level,
            smooth_sigma=args.smooth_sigma,
            train_fraction=args.train_fraction,
        )


if __name__ == "__main__":
    main()
