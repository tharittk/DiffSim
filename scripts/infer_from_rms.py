#!/usr/bin/env python
"""
Inference script: Generate facies realizations from real seismic RMS + sparse wells.

Takes a real RMS amplitude map and sparse well facies observations as input,
and generates multiple plausible facies realizations using the trained
conditional diffusion model.

Input formats:
    - RMS map: 2D numpy array (.npy) or image (.png/.tif) normalized to [-1, 1]
    - Well data: CSV file with columns [row, col, facies_code]
      where facies_code is {0: mud, 1: bank, 2: sand}

Output:
    - Multiple facies realizations as .npy and/or .png files
    - Mean probability maps for each facies type
    - Summary statistics

Usage:
    python scripts/infer_from_rms.py \
        --config configs/case1_flumy.json \
        --checkpoint model/case1_flumy_conditional/best_model.pth \
        --rms_map data/real/rms_amplitude.npy \
        --wells data/real/well_facies.csv \
        --output_dir results/real_inference \
        --n_realizations 50

    # Or with a pre-made conditioning .npz (if you already have prepared input)
    python scripts/infer_from_rms.py \
        --config configs/case1_flumy.json \
        --checkpoint model/case1_flumy_conditional/best_model.pth \
        --prepared_input data/real/prepared_input.npz \
        --output_dir results/real_inference
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffsim.core.network import Network
from diffsim.data.flumy_generator import (
    FACIES_MUD,
    FACIES_BANK,
    FACIES_SAND,
    denormalize_facies,
)
from diffsim.data.well_sampling import create_well_mask, create_well_conditioning


def load_rms_map(rms_path, image_size):
    """
    Load and normalize an RMS amplitude map.

    Supports .npy (numpy), .npz, and image formats (.png, .tif, .jpg).
    Normalizes to [-1, 1] range.

    Args:
        rms_path: Path to RMS map file
        image_size: Target (H, W) for resizing

    Returns:
        2D numpy array (H, W) in [-1, 1]
    """
    rms_path = Path(rms_path)
    ext = rms_path.suffix.lower()

    if ext == ".npy":
        rms = np.load(rms_path)
    elif ext == ".npz":
        data = np.load(rms_path)
        # Try common key names
        for key in ["rms", "data", "amplitude", "map"]:
            if key in data:
                rms = data[key]
                break
        else:
            # Use first key
            rms = data[list(data.keys())[0]]
    elif ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg"):
        from PIL import Image

        img = Image.open(rms_path).convert("F")
        rms = np.array(img, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported RMS format: {ext}")

    # Resize if needed
    if rms.shape != tuple(image_size):
        from PIL import Image

        img = Image.fromarray(rms.astype(np.float32), mode="F")
        img = img.resize((image_size[1], image_size[0]), Image.BILINEAR)
        rms = np.array(img, dtype=np.float32)

    # Normalize to [-1, 1]
    rms_min, rms_max = rms.min(), rms.max()
    if rms_max - rms_min > 1e-10:
        rms = 2.0 * (rms - rms_min) / (rms_max - rms_min) - 1.0
    else:
        rms = np.zeros_like(rms)

    return rms.astype(np.float32)


def load_well_data(well_path, image_size):
    """
    Load well data from CSV file.

    Expected CSV format (with or without header):
        row, col, facies_code

    Where facies_code is {0: mud, 1: bank, 2: sand}

    Args:
        well_path: Path to CSV file
        image_size: (H, W) for boundary validation

    Returns:
        well_positions: (N, 2) array of (row, col)
        facies_at_wells: (N,) array of facies codes
    """
    import csv

    well_path = Path(well_path)
    positions = []
    facies_codes = []

    with open(well_path, "r") as f:
        reader = csv.reader(f)
        for i, row_data in enumerate(reader):
            if not row_data or len(row_data) < 3:
                continue
            try:
                r, c, fac = int(row_data[0]), int(row_data[1]), int(row_data[2])
            except ValueError:
                if i == 0:
                    # Skip header row
                    continue
                raise ValueError(
                    f"Could not parse row {i} in {well_path}: {row_data}"
                )

            # Validate bounds
            h, w = image_size
            if 0 <= r < h and 0 <= c < w and fac in (0, 1, 2):
                positions.append((r, c))
                facies_codes.append(fac)
            else:
                print(
                    f"  Warning: Skipping invalid well at row={r}, col={c}, "
                    f"facies={fac} (image_size={image_size})"
                )

    if len(positions) == 0:
        raise ValueError(f"No valid wells found in {well_path}")

    return np.array(positions, dtype=np.int64), np.array(facies_codes, dtype=np.int8)


def build_conditioning(rms, well_positions, facies_at_wells, image_size):
    """
    Build the 5-channel conditioning tensor for the model.

    Channels: [rms, well_presence, sand, bank, mud]

    Args:
        rms: (H, W) RMS amplitude map normalized to [-1, 1]
        well_positions: (N, 2) array of (row, col)
        facies_at_wells: (N,) array of facies codes at well locations
        image_size: (H, W)

    Returns:
        cond_image: (5, H, W) tensor
        mask: (1, H, W) tensor (0 at wells, 1 elsewhere)
    """
    h, w = image_size

    # Build a synthetic facies_map with values only at well locations
    facies_map = np.zeros((h, w), dtype=np.int8)
    for (r, c), fac in zip(well_positions, facies_at_wells):
        facies_map[r, c] = fac

    # Well conditioning: [well_presence, sand, bank, mud]
    well_cond = create_well_conditioning(facies_map, well_positions)

    # Full conditioning: [rms, well_presence, sand, bank, mud]
    rms_channel = rms[np.newaxis, :, :]  # (1, H, W)
    cond_image = np.concatenate([rms_channel, well_cond], axis=0)  # (5, H, W)

    # Diffusion mask: 0 at wells, 1 elsewhere
    mask = create_well_mask(image_size, well_positions)

    return (
        torch.from_numpy(cond_image).float(),
        torch.from_numpy(mask[np.newaxis, :, :]).float(),
    )


def facies_to_color(facies_int):
    """Convert integer facies map to RGB color image for visualization."""
    h, w = facies_int.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[facies_int == FACIES_MUD] = [139, 90, 43]  # brown for mud
    color[facies_int == FACIES_BANK] = [144, 238, 144]  # light green for bank
    color[facies_int == FACIES_SAND] = [255, 255, 0]  # yellow for sand
    return color


def run_inference(
    config_path,
    checkpoint_path,
    rms_map_path=None,
    well_path=None,
    prepared_input_path=None,
    output_dir="results/inference",
    n_realizations=50,
    use_ddim=True,
    ddim_steps=50,
    device_str="cuda",
):
    """
    Run multi-realization inference on real data.

    Args:
        config_path: Path to config JSON
        checkpoint_path: Path to trained model checkpoint
        rms_map_path: Path to RMS amplitude map
        well_path: Path to well CSV file
        prepared_input_path: Path to pre-made .npz with 'cond' and 'mask' arrays
        output_dir: Output directory
        n_realizations: Number of facies realizations to generate
        use_ddim: Use DDIM (faster) or DDPM (higher quality)
        ddim_steps: Number of DDIM steps (if use_ddim=True)
        device_str: Device string ('cuda', 'cpu', 'cuda:0', etc.)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    cond = config["conditional"]
    image_size = config["image_size"]
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    # Build network
    unet_config = {
        "image_size": image_size[0],
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
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        network.load_state_dict(checkpoint["model_state"])
    elif isinstance(checkpoint, dict) and "ema_state" in checkpoint:
        network.load_state_dict(checkpoint["ema_state"])
    else:
        network.load_state_dict(checkpoint)

    network.to(device)
    network.eval()

    # Set test noise schedule
    network.set_new_noise_schedule(device=device, phase="test")

    # Build conditioning
    if prepared_input_path:
        data = np.load(prepared_input_path)
        cond_image = torch.from_numpy(data["cond"]).float()
        mask = torch.from_numpy(data["mask"]).float()
        if cond_image.dim() == 3:
            cond_image = cond_image.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)
    else:
        if rms_map_path is None or well_path is None:
            raise ValueError(
                "Either --prepared_input or both --rms_map and --wells must be provided"
            )
        rms = load_rms_map(rms_map_path, image_size)
        well_positions, facies_at_wells = load_well_data(well_path, image_size)
        cond_image, mask = build_conditioning(
            rms, well_positions, facies_at_wells, image_size
        )
        cond_image = cond_image.unsqueeze(0)  # (1, 5, H, W)
        mask = mask.unsqueeze(0)  # (1, 1, H, W)

        # Save the prepared input for future re-use
        np.savez_compressed(
            output_dir / "prepared_input.npz",
            cond=cond_image.squeeze(0).numpy(),
            mask=mask.squeeze(0).numpy(),
            rms=rms,
            well_positions=well_positions,
            facies_at_wells=facies_at_wells,
        )

    cond_image = cond_image.to(device)
    mask = mask.to(device)

    # Build a dummy y_0 for inpainting (from well facies)
    # At well locations: known normalized facies. Elsewhere: 0 (doesn't matter, mask=1)
    y_0 = torch.zeros(1, 1, image_size[0], image_size[1], device=device)
    if not prepared_input_path:
        from diffsim.data.flumy_generator import normalize_facies

        facies_map_dummy = np.zeros(image_size, dtype=np.int8)
        for (r, c), fac in zip(well_positions, facies_at_wells):
            facies_map_dummy[r, c] = fac
        y_0_np = normalize_facies(facies_map_dummy)
        y_0 = torch.from_numpy(y_0_np[np.newaxis, np.newaxis, :, :]).float().to(device)

    print(f"=== Inference from Real Data ===")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Image size: {image_size}")
    print(f"Conditioning channels: {cond_image.shape}")
    print(f"Realizations: {n_realizations}")
    print(f"Method: {'DDIM' if use_ddim else 'DDPM'}")
    if use_ddim:
        print(f"DDIM steps: {ddim_steps}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print()

    # Generate realizations
    all_realizations = []

    with torch.no_grad():
        for i in tqdm(range(n_realizations), desc="Generating realizations"):
            if use_ddim:
                output, _ = network.restoration_ddim(
                    cond_image,
                    y_0=y_0,
                    mask=mask,
                    ddim_steps=ddim_steps,
                    eta=0.0,
                    sample_num=4,
                )
            else:
                output, _ = network.restoration(
                    cond_image,
                    y_0=y_0,
                    mask=mask,
                    sample_num=4,
                )

            # Convert to facies codes
            output_np = output.squeeze().cpu().numpy()
            facies_int = denormalize_facies(output_np)
            all_realizations.append(facies_int)

            # Save individual realization
            np.save(output_dir / f"realization_{i:03d}.npy", facies_int)

            # Save as color image
            color_img = facies_to_color(facies_int)
            from PIL import Image

            Image.fromarray(color_img).save(
                output_dir / f"realization_{i:03d}.png"
            )

    all_realizations = np.stack(all_realizations, axis=0)  # (N, H, W)

    # Compute probability maps
    print("\nComputing probability maps...")
    sand_prob = np.mean(all_realizations == FACIES_SAND, axis=0)
    bank_prob = np.mean(all_realizations == FACIES_BANK, axis=0)
    mud_prob = np.mean(all_realizations == FACIES_MUD, axis=0)

    # Most likely facies (mode)
    mode_facies = np.zeros(image_size, dtype=np.int8)
    mode_facies[bank_prob >= np.maximum(sand_prob, mud_prob)] = FACIES_BANK
    mode_facies[sand_prob >= np.maximum(bank_prob, mud_prob)] = FACIES_SAND
    mode_facies[mud_prob >= np.maximum(sand_prob, bank_prob)] = FACIES_MUD

    # Entropy (uncertainty measure)
    probs = np.stack([mud_prob, bank_prob, sand_prob], axis=-1)
    probs = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs * np.log2(probs), axis=-1)

    # Save probability maps
    np.savez_compressed(
        output_dir / "probability_maps.npz",
        sand_prob=sand_prob,
        bank_prob=bank_prob,
        mud_prob=mud_prob,
        mode_facies=mode_facies,
        entropy=entropy,
        all_realizations=all_realizations,
    )

    # Save probability map images
    from PIL import Image

    for name, prob_map in [
        ("sand_probability", sand_prob),
        ("bank_probability", bank_prob),
        ("mud_probability", mud_prob),
        ("entropy", entropy / np.log2(3)),  # Normalize to [0, 1]
    ]:
        prob_uint8 = (prob_map * 255).astype(np.uint8)
        Image.fromarray(prob_uint8).save(output_dir / f"{name}.png")

    # Save mode facies
    Image.fromarray(facies_to_color(mode_facies)).save(
        output_dir / "mode_facies.png"
    )

    # Print summary
    print(f"\n=== Results Summary ===")
    print(f"Realizations saved: {n_realizations}")
    print(f"Mean sand fraction: {sand_prob.mean():.3f}")
    print(f"Mean bank fraction: {bank_prob.mean():.3f}")
    print(f"Mean mud fraction:  {mud_prob.mean():.3f}")
    print(f"Mean entropy (normalized): {(entropy / np.log2(3)).mean():.3f}")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate facies realizations from real RMS + well data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to config JSON"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth)",
    )

    # Input data
    input_group = parser.add_argument_group("Input Data")
    input_group.add_argument(
        "--rms_map",
        type=str,
        default=None,
        help="Path to RMS amplitude map (.npy, .npz, or image)",
    )
    input_group.add_argument(
        "--wells",
        type=str,
        default=None,
        help="Path to well data CSV (columns: row, col, facies_code)",
    )
    input_group.add_argument(
        "--prepared_input",
        type=str,
        default=None,
        help="Path to pre-made conditioning .npz (alternative to --rms_map + --wells)",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/real_inference",
        help="Output directory",
    )

    # Inference settings
    parser.add_argument(
        "--n_realizations",
        type=int,
        default=50,
        help="Number of facies realizations to generate",
    )
    parser.add_argument(
        "--use_ddim",
        action="store_true",
        default=True,
        help="Use DDIM for faster sampling (default: True)",
    )
    parser.add_argument(
        "--use_ddpm",
        action="store_true",
        default=False,
        help="Use DDPM for higher quality (overrides --use_ddim)",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="Number of DDIM steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device ('cuda', 'cpu', 'cuda:0')",
    )

    args = parser.parse_args()

    use_ddim = not args.use_ddpm

    run_inference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        rms_map_path=args.rms_map,
        well_path=args.wells,
        prepared_input_path=args.prepared_input,
        output_dir=args.output_dir,
        n_realizations=args.n_realizations,
        use_ddim=use_ddim,
        ddim_steps=args.ddim_steps,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
