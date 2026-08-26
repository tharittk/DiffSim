"""Run tiled conditional facies inference for the h05 RMS survey."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from tqdm.auto import tqdm


# Locate the repository when launched from style-matching or the repository root.
repo_root = Path.cwd().resolve()
if not (repo_root / "diffsim").exists():
    candidate = repo_root.parent
    if (candidate / "diffsim").exists():
        os.chdir(candidate)
        repo_root = candidate

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from diffsim.core.network import Network
from diffsim.data.flumy_generator import denormalize_lithofacies


# Paths and settings
RUN_TIMESTAMP = "20260818_102941"
MODEL_DIR = Path("/mnt/sda_data/tharitt/diffsim/model/case1_flumy_conditional") / RUN_TIMESTAMP
CONFIG_PATH = MODEL_DIR / "config.json"
OUTPUT_DIR = Path("/mnt/sda_data/tharitt/diffsim/results") / f"arthit_inference_{RUN_TIMESTAMP}"
H05_PATH = repo_root / "style-matching" / "h05_sub9"

if not MODEL_DIR.is_dir():
    raise FileNotFoundError(f"Model run directory not found: {MODEL_DIR}")
if not CONFIG_PATH.is_file():
    raise FileNotFoundError(f"Model config not found: {CONFIG_PATH}")
if not H05_PATH.is_file():
    raise FileNotFoundError(f"h05 input file not found: {H05_PATH}")

CHECKPOINT_PATH = MODEL_DIR / "best_model.pth"
if not CHECKPOINT_PATH.exists():
    candidates = sorted(MODEL_DIR.glob("checkpoint_epoch_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {MODEL_DIR}")
    CHECKPOINT_PATH = candidates[-1]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 64
PATCH_STRIDE = 64
SUBSAMPLE_FACTOR = 4
CLIP_PERCENTILE = 99.5
N_SAMPLES = 8
DDIM_STEPS = 100
ETA = 0.0
BATCH_SIZE = 16
RUN_FULL_MAP = True
SMOKE_MAX_PATCHES = 20
SEED = 88

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("repo_root =", repo_root)
print("=== FILE PARAMS:===")
print("RUN_TIMESTAMP  :", RUN_TIMESTAMP)
print("CONFIG_PATH    :", CONFIG_PATH)
print("CHECKPOINT_PATH:", CHECKPOINT_PATH)
print("H05_PATH       :", H05_PATH)
print("OUTPUT_DIR     :", OUTPUT_DIR)
print("DEVICE         :", DEVICE)
print("BATCH_SIZE     :", BATCH_SIZE)
print("===INFERENCE PARAMS===")
print("SUBSAMPLE      :", SUBSAMPLE_FACTOR)
print("N_SAMPLES      :", N_SAMPLES)
print("DDIM_STEPS     :", DDIM_STEPS)
print("ETA            :", ETA)
print("Mode           :", "full h05 map" if RUN_FULL_MAP else f"smoke test ({SMOKE_MAX_PATCHES} patches)")


def load_model(config_path: Path, checkpoint_path: Path, device: str):
    with open(config_path) as handle:
        config = json.load(handle)

    conditional = config["conditional"]
    unet_config = {
        "image_size": config["image_size"],
        "in_channel": conditional["in_channel"],
        "out_channel": conditional["out_channel"],
        "inner_channel": conditional["inner_channel"],
        "channel_mults": conditional["channel_mults"],
        "attn_res": conditional["attn_res"],
        "res_blocks": conditional["res_blocks"],
        "dropout": conditional["dropout"],
    }

    network = Network(
        unet=unet_config,
        beta_schedule=conditional["beta_schedule"],
        module_name=conditional.get("module_name", "guided_diffusion"),
        predict_type=conditional.get("predict_type", "epsilon"),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    incompatible = network.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print("Missing keys:", incompatible.missing_keys)
    if incompatible.unexpected_keys:
        print("Unexpected keys (ignored):", incompatible.unexpected_keys)

    network.to(device)
    network.set_new_noise_schedule(device=torch.device(device), phase="test")
    network.eval()
    return network, config


def load_h05_grid(path):
    data = np.loadtxt(path, comments="#", usecols=(0, 1, 2, 3, 4), dtype=np.float64)
    x, y, amplitude = data[:, 0], data[:, 1], data[:, 2].astype(np.float32)
    columns, rows = data[:, 3].astype(np.int32), data[:, 4].astype(np.int32)
    row_min, row_max = int(rows.min()), int(rows.max())
    col_min, col_max = int(columns.min()), int(columns.max())
    shape = (row_max - row_min + 1, col_max - col_min + 1)
    grid = np.full(shape, np.nan, dtype=np.float32)
    grid[rows - row_min, columns - col_min] = amplitude

    design = np.column_stack([np.ones(len(rows)), columns, rows])
    x_affine = np.linalg.lstsq(design, x, rcond=None)[0]
    y_affine = np.linalg.lstsq(design, y, rcond=None)[0]
    return grid, {
        "row_min": row_min,
        "row_max": row_max,
        "col_min": col_min,
        "col_max": col_max,
        "x_affine": x_affine,
        "y_affine": y_affine,
    }


def subsample_grid(grid, georef, factor):
    """Block-average the grid by `factor` (NaN-aware) and rebase the georeference affines."""
    if factor <= 1:
        return grid, georef

    height, width = grid.shape
    padded_height = int(np.ceil(height / factor)) * factor
    padded_width = int(np.ceil(width / factor)) * factor
    padded = np.full((padded_height, padded_width), np.nan, dtype=np.float32)
    padded[:height, :width] = grid

    blocks = padded.reshape(padded_height // factor, factor, padded_width // factor, factor)
    valid = np.isfinite(blocks)
    totals = np.where(valid, blocks, 0.0).sum(axis=(1, 3))
    counts = valid.sum(axis=(1, 3))
    coarse = np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0).astype(np.float32)

    offset = (factor - 1) / 2.0
    x_affine, y_affine = georef["x_affine"], georef["y_affine"]
    row_min, col_min = georef["row_min"], georef["col_min"]

    def rebase(affine):
        constant, per_col, per_row = affine
        return np.array(
            [
                constant + per_col * (col_min + offset) + per_row * (row_min + offset),
                per_col * factor,
                per_row * factor,
            ]
        )

    coarse_georef = dict(georef)
    coarse_georef.update(
        {
            "row_min": 0,
            "col_min": 0,
            "row_max": coarse.shape[0] - 1,
            "col_max": coarse.shape[1] - 1,
            "x_affine": rebase(x_affine),
            "y_affine": rebase(y_affine),
            "subsample_factor": factor,
        }
    )
    return coarse, coarse_georef


def patch_origins(length, stride):
    return list(range(0, length, stride))


def extract_edge_safe_patch(grid, row0, col0, patch_size=64):
    height, width = grid.shape
    row1, col1 = min(row0 + patch_size, height), min(col0 + patch_size, width)
    patch = np.full((patch_size, patch_size), np.nan, dtype=np.float32)
    patch[: row1 - row0, : col1 - col0] = grid[row0:row1, col0:col1]
    valid = np.isfinite(patch)
    if not valid.any():
        return None

    nearest_indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = patch[tuple(nearest_indices)]
    observed = patch[valid]
    value_min, value_max = float(observed.min()), float(observed.max())
    if value_max - value_min < 1e-10:
        normalized = np.zeros_like(filled, dtype=np.float32)
    else:
        normalized = (2.0 * (filled - value_min) / (value_max - value_min) - 1.0).astype(np.float32)
    return normalized, valid, (row1, col1)


def build_tiles(grid, patch_size=64, stride=64):
    tiles = []
    for row0 in patch_origins(grid.shape[0], stride):
        for col0 in patch_origins(grid.shape[1], stride):
            extracted = extract_edge_safe_patch(grid, row0, col0, patch_size)
            if extracted is not None:
                tiles.append((row0, col0, *extracted))
    return tiles


@torch.no_grad()
def infer_batch(network, normalized_patches, n_samples, ddim_steps, eta, device):
    conditioning = torch.from_numpy(np.stack(normalized_patches)[:, None]).float().to(device)
    counts = np.zeros((len(normalized_patches), 3, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    for _ in range(n_samples):
        generated, _ = network.restoration_ddim(
            y_cond=conditioning,
            ddim_steps=ddim_steps,
            eta=eta,
            sample_num=min(4, ddim_steps - 1),
        )
        facies = denormalize_lithofacies(generated[:, 0].cpu().numpy())
        for code in range(3):
            counts[:, code] += facies == code
    return counts / n_samples


def infer_tiles(network, tiles, output_dir, batch_size=16):
    cache_label = f"{H05_PATH.name}_sub{SUBSAMPLE_FACTOR}_samples{N_SAMPLES}_steps{DDIM_STEPS}_eta{ETA:g}"
    tile_dir = output_dir / "tiles" / cache_label
    tile_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for start in tqdm(range(0, len(tiles), batch_size), desc="Tile batches"):
        batch_tiles = tiles[start : start + batch_size]
        batch_probabilities = [None] * len(batch_tiles)
        pending_indices, pending_patches = [], []
        for index, (row0, col0, normalized, valid, bounds) in enumerate(batch_tiles):
            tile_path = tile_dir / f"tile_sub{SUBSAMPLE_FACTOR}_r{row0:04d}_c{col0:04d}.npz"
            if tile_path.exists():
                with np.load(tile_path) as cached:
                    batch_probabilities[index] = cached["probabilities"]
            else:
                pending_indices.append(index)
                pending_patches.append(normalized)
        if pending_patches:
            inferred = infer_batch(network, pending_patches, N_SAMPLES, DDIM_STEPS, ETA, DEVICE)
            for index, probabilities in zip(pending_indices, inferred):
                row0, col0, _, valid, bounds = batch_tiles[index]
                np.savez_compressed(
                    tile_dir / f"tile_sub{SUBSAMPLE_FACTOR}_r{row0:04d}_c{col0:04d}.npz",
                    probabilities=probabilities,
                    valid=valid,
                    bounds=np.asarray(bounds),
                )
                batch_probabilities[index] = probabilities
        results.extend(zip(batch_tiles, batch_probabilities))
    return results


def reconstruct(shape, inferred_tiles):
    probability_sum = np.zeros((3, *shape), dtype=np.float32)
    weights = np.zeros(shape, dtype=np.float32)
    for tile, probabilities in inferred_tiles:
        row0, col0, _, valid, (row1, col1) = tile
        tile_height, tile_width = row1 - row0, col1 - col0
        local_valid = valid[:tile_height, :tile_width]
        probability_sum[:, row0:row1, col0:col1] += (
            probabilities[:, :tile_height, :tile_width] * local_valid[None]
        )
        weights[row0:row1, col0:col1] += local_valid

    covered = weights > 0
    probability_maps = np.full((3, *shape), np.nan, dtype=np.float32)
    probability_maps[:, covered] = probability_sum[:, covered] / weights[covered]
    most_likely = np.full(shape, -1, dtype=np.int8)
    most_likely[covered] = np.argmax(probability_maps[:, covered], axis=0).astype(np.int8)
    return probability_maps, most_likely, weights


def main():
    network, config = load_model(CONFIG_PATH, CHECKPOINT_PATH, DEVICE)
    print(f"Loaded: {CHECKPOINT_PATH}")
    print(f"Training image size: {config['image_size']}")
    print(f"UNet channels: {config['conditional']['in_channel']} -> {config['conditional']['out_channel']}")
    print(f"Prediction target: {config['conditional'].get('predict_type', 'epsilon')}")

    rms_grid, georef = load_h05_grid(H05_PATH)
    if SUBSAMPLE_FACTOR > 1:
        original_shape = rms_grid.shape
        rms_grid, georef = subsample_grid(rms_grid, georef, SUBSAMPLE_FACTOR)
        print(f"Subsampled {original_shape} -> {rms_grid.shape} (factor {SUBSAMPLE_FACTOR})")

    valid_grid = np.isfinite(rms_grid)
    if not valid_grid.any():
        raise RuntimeError(f"No valid RMS values found in {H05_PATH}")

    rms_clipped = rms_grid.copy()
    valid_values = rms_grid[valid_grid]
    clip_high = float(np.percentile(valid_values, CLIP_PERCENTILE))
    rms_clipped[valid_grid] = np.clip(valid_values, valid_values.min(), clip_high)

    print(f"Grid shape: {rms_grid.shape} (rows x cols)")
    print(f"Observed cells: {valid_grid.sum():,} / {valid_grid.size:,}")
    print(f"Amplitude clip: min={valid_values.min():.3f}, p{CLIP_PERCENTILE}={clip_high:.3f}")

    all_tiles = build_tiles(rms_clipped, PATCH_SIZE, PATCH_STRIDE)
    if not all_tiles:
        raise RuntimeError("No tiles available for inference.")
    if RUN_FULL_MAP:
        tiles = all_tiles
    else:
        regular_count = max(0, min(SMOKE_MAX_PATCHES - 1, len(all_tiles) - 1))
        rng = np.random.RandomState(SEED)
        shuffled_tiles = all_tiles.copy()
        rng.shuffle(shuffled_tiles)
        tiles = shuffled_tiles[:regular_count] + [all_tiles[-1]]

    print(f"Valid tiles: {len(all_tiles):,}; selected: {len(tiles):,}")
    print(f"Last selected tile origin: {tiles[-1][0:2]}, valid pixels: {tiles[-1][3].sum():,}")

    inferred_tiles = infer_tiles(network, tiles, OUTPUT_DIR, BATCH_SIZE)
    print(f"Inferred or resumed {len(inferred_tiles)} tiles")

    probability_maps, most_likely, weights = reconstruct(rms_grid.shape, inferred_tiles)
    covered = weights > 0
    print(f"Covered observed pixels: {covered.sum():,}")
    print("Predicted facies counts:", dict(zip(*np.unique(most_likely[covered], return_counts=True))))

    synthetic = np.full((70, 75), np.nan, dtype=np.float32)
    synthetic[2:68, 3:73] = np.arange(66 * 70, dtype=np.float32).reshape(66, 70)
    synthetic[30:35, 30:35] = np.nan
    synthetic_tiles = build_tiles(synthetic, patch_size=64, stride=64)
    bottom_right = next(tile for tile in synthetic_tiles if tile[0] == 64 and tile[1] == 64)
    row0, col0, normalized, valid, (row1, col1) = bottom_right
    assert normalized.shape == (64, 64)
    assert valid.shape == (64, 64)
    assert (row1, col1) == synthetic.shape
    assert row1 <= synthetic.shape[0] and col1 <= synthetic.shape[1]
    assert valid[: row1 - row0, : col1 - col0].any()
    assert not valid[row1 - row0 :, :].any()
    assert not valid[:, col1 - col0 :].any()
    assert np.isfinite(normalized).all()
    if RUN_FULL_MAP:
        assert np.array_equal(covered, valid_grid)
    else:
        assert np.all(covered <= valid_grid)
    assert most_likely[~covered].max(initial=-1) == -1
    assert np.isfinite(probability_maps[:, covered]).all()
    assert np.allclose(probability_maps[:, covered].sum(axis=0), 1.0, atol=1e-5)
    print("Edge tests passed: partial borders, interior holes, bounds, masks, and probability sums are valid.")

    mode_label = "full" if RUN_FULL_MAP else "smoke"
    sampling_label = f"samples{N_SAMPLES}_steps{DDIM_STEPS}_eta{ETA:g}_sub{SUBSAMPLE_FACTOR}"
    output_path = OUTPUT_DIR / f"arthit_inference_{mode_label}_{sampling_label}.npz"
    np.savez_compressed(
        output_path,
        probabilities=probability_maps,
        most_likely=most_likely,
        row_min=georef["row_min"],
        col_min=georef["col_min"],
        x_affine=georef["x_affine"],
        y_affine=georef["y_affine"],
        facies_names=np.asarray(["shale", "sand", "silt"]),
        subsample_factor=SUBSAMPLE_FACTOR,
        n_samples=N_SAMPLES,
        ddim_steps=DDIM_STEPS,
        eta=ETA,
    )
    print(f"Saved: {output_path}")
    print("Inference complete.")


if __name__ == "__main__":
    main()
