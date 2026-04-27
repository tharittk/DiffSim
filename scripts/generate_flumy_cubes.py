#!/usr/bin/env python
"""
Generate 3D facies cubes using Flumy for a single (ng, isbx) combination.

Usage:
    python scripts/generate_flumy_cubes.py \
        --ng 50 --isbx 80 --n_seeds 10 --output_dir data/flumy3d/facies

This is called by scripts/generate_all_cubes.sh which iterates over
all (ng, isbx) combinations. The resulting .npy files are immutable
Flumy outputs — generated once and frozen.

Output naming:  ng{ng}_isbx{isbx}_seed{seed}.npy
Each file is a 3D int8 array of shape (nx, ny, nz) with facies codes {0,1,2}.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffsim.data.flumy_generator import FlumyGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 3D Flumy facies cubes for a single (ng, isbx) combo."
    )
    parser.add_argument("--nx", type=int, default=256, help="Grid nodes along x")
    parser.add_argument("--ny", type=int, default=256, help="Grid nodes along y")
    parser.add_argument("--mesh", type=int, default=20, help="Grid mesh size (m)")
    parser.add_argument("--hmax", type=float, default=5.0, help="Max channel depth (m)")
    parser.add_argument("--ng", type=int, required=True, help="Net-to-gross (0-100)")
    parser.add_argument("--isbx", type=int, required=True, help="Sand body extension")
    parser.add_argument("--dz", type=float, default=1.0, help="Vertical step (m)")
    parser.add_argument(
        "--zul", type=float, default=None, help="Reservoir height (m), default=3*hmax"
    )
    parser.add_argument(
        "--n_seeds", type=int, default=10, help="Number of seeds (realizations)"
    )
    parser.add_argument(
        "--seed_start",
        type=int,
        default=1,
        help="Starting seed value (Flumy requires >= 1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/flumy3d/facies",
        help="Output directory for .npy files",
    )
    parser.add_argument("--verbose", action="store_true", help="Flumy verbose output")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = FlumyGenerator(
        nx=args.nx,
        ny=args.ny,
        mesh=args.mesh,
        hmax=args.hmax,
        ng=args.ng,
        isbx=args.isbx,
        zul=args.zul,
        dz=args.dz,
        verbose=args.verbose,
    )

    for i in range(args.n_seeds):
        seed = args.seed_start + i
        fname = f"ng{args.ng}_isbx{args.isbx}_seed{seed}.npy"
        out_path = output_dir / fname

        if out_path.exists():
            print(f"  [skip] {out_path} already exists")
            continue

        t0 = time.time()
        try:
            facies_raw = gen.generate(seed * args.ng * args.isbx)
            np.save(out_path, facies_raw)
            elapsed = time.time() - t0
            print(f"  [done] {fname}  shape={facies_raw.shape}  {elapsed:.1f}s")
        except RuntimeError as e:
            print(f"  [FAIL] seed={seed}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
