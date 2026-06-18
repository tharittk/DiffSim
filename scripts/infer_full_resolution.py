#!/usr/bin/env python
"""
Run inference on full-resolution (256x256) RMS maps using a model trained on 64x64.

The fully-convolutional UNet generalizes to arbitrary input sizes at inference
time. This script loads a trained checkpoint, runs multiple stochastic inferences,
and outputs per-facies probability maps.

Usage:
    python scripts/infer_full_resolution.py \
        --config configs/case1_flumy.json \
        --rms_dir /path/to/rms_256x256/ \
        --output_dir results/inference_256 \
        --n_samples 20 \
        --ddim_steps 50

    # Or single file:
    python scripts/infer_full_resolution.py \
        --config configs/case1_flumy.json \
        --rms_file /path/to/single_rms.npy \
        --output_dir results/inference_256 \
        --n_samples 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffsim.core.network import Network
from diffsim.core.inference import inference_full_resolution


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-resolution inference with probability output."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config JSON (for model architecture)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint. If None, uses config checkpoints.conditional",
    )
    parser.add_argument(
        "--rms_dir",
        type=str,
        default=None,
        help="Directory with 2D RMS .npy files (H, W) at full resolution",
    )
    parser.add_argument(
        "--rms_file",
        type=str,
        default=None,
        help="Single RMS .npy file to run inference on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/inference_256",
        help="Output directory for probability maps and predictions",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of stochastic inferences per input",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="Number of DDIM sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.3,
        help="DDIM stochasticity (>0 for diverse ensemble members)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    return parser.parse_args()


def load_model(config_path, checkpoint_path, device):
    """Load trained network from config and checkpoint."""
    with open(config_path) as f:
        config = json.load(f)

    cond = config["conditional"]

    # Build UNet config - use a dummy image_size; the model is fully convolutional
    unet_config = {
        "image_size": config["image_size"],
        "in_channel": cond["in_channel"],
        "out_channel": cond["out_channel"],
        "inner_channel": cond["inner_channel"],
        "channel_mults": cond["channel_mults"],
        "attn_res": cond["attn_res"],
        "res_blocks": cond["res_blocks"],
        "dropout": 0.0,  # No dropout at inference
    }

    module_name = cond.get("module_name", "guided_diffusion")
    predict_type = cond.get("predict_type", "epsilon")

    network = Network(
        unet=unet_config,
        beta_schedule=cond["beta_schedule"],
        module_name=module_name,
        predict_type=predict_type,
    )

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = config.get("checkpoints", {}).get("conditional")
    if checkpoint_path is None:
        raise ValueError("No checkpoint path specified in args or config.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        network.load_state_dict(checkpoint["model_state"])
    else:
        network.load_state_dict(checkpoint)

    network.to(device)
    network.eval()
    return network


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # Collect input files
    rms_files = []
    if args.rms_file:
        rms_files = [Path(args.rms_file)]
    elif args.rms_dir:
        rms_files = sorted(Path(args.rms_dir).glob("*.npy"))
    else:
        print("Error: provide --rms_dir or --rms_file", file=sys.stderr)
        sys.exit(1)

    if not rms_files:
        print("Error: no .npy files found.", file=sys.stderr)
        sys.exit(1)

    # Load model
    print(f"Loading model from config: {args.config}")
    network = load_model(args.config, args.checkpoint, device)
    print(f"Model loaded. Running {args.n_samples} samples per input, DDIM steps={args.ddim_steps}")
    print(f"Input files: {len(rms_files)}")
    print()

    for i, rms_path in enumerate(rms_files, 1):
        t0 = time.time()
        rms_map = np.load(rms_path)

        if rms_map.ndim != 2:
            print(f"  [{i}] Skipping {rms_path.name}: expected 2D, got {rms_map.ndim}D")
            continue

        print(f"  [{i}/{len(rms_files)}] {rms_path.name} shape={rms_map.shape}")

        probabilities, samples, most_likely = inference_full_resolution(
            network,
            rms_map,
            n_samples=args.n_samples,
            ddim_steps=args.ddim_steps,
            eta=args.eta,
            device=device,
        )

        # Save results
        stem = rms_path.stem
        np.save(output_dir / f"{stem}_most_likely.npy", most_likely)
        np.save(output_dir / f"{stem}_samples.npy", samples)
        for code, prob_map in probabilities.items():
            np.save(output_dir / f"{stem}_prob_facies{code}.npy", prob_map)

        elapsed = time.time() - t0
        print(f"      → saved probabilities + {args.n_samples} samples  ({elapsed:.1f}s)")

    print("\nDone. Results saved to:", output_dir)


if __name__ == "__main__":
    main()
