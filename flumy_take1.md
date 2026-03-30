# Flumy Data Generation Pipeline — Take 1

## Overview

This document summarizes the initial implementation of a Flumy-based training data generation pipeline for the DiffSim project. The goal is to replace the original Petrel-generated synthetic dataset with data from the [Flumy](https://flumy.minesparis.psl.eu/) Python package, enabling large-scale, reproducible training data generation entirely in Python.

## Motivation (from plan.md)

The original DiffSim repository trains a conditional diffusion model to predict seismic facies from seismic attributes. The original training data was generated in Petrel (proprietary software) and stored as grayscale PNG images + binary mask PNGs. We want to:

1. Generate training data programmatically using Flumy (open-source process-based channelized reservoir simulator)
2. Compute synthetic RMS amplitude attributes via seismic forward modeling (physically rigorous)
3. Sample sparse well facies from the true facies maps (random column locations)
4. Train the same conditional diffusion architecture on this new data

The scope is restricted to **2D channel facies (Case 1)** only. Mud drape (Case 2) and 3D (Case 3) are ignored.

---

## What Changed from the Original Repository

### Original Architecture (unchanged)

The core diffusion model code is **untouched**:
- `diffsim/core/network.py` — Conditional diffusion Network class
- `diffsim/core/diffusion.py` — Unconditional diffusion
- `diffsim/models/` — UNet architectures (2D and 3D)
- `diffsim/models/guided_diffusion/` — Guided diffusion UNets

### Original Data Flow (Case 1)

```
Petrel → grayscale PNG images (facies)
       → binary mask PNGs (well locations)
       → InpaintDatasetCase1 loads both, creates 5-channel conditioning:
         [known_mask, sand_indicator, bank_indicator, mud_indicator, y_t]
       → in_channel = 5, out_channel = 1
```

**Key details of original approach:**
- Facies stored as grayscale PNGs, normalized via `transforms.Normalize(mean=0.5, std=0.5)` to [-1, 1]
- Sand ≈ 1.0, Bank ≈ -0.004 (gray value ~127), Mud ≈ -1.0
- Masks loaded from separate PNG files (binary: well locations)
- Dataset selects `InpaintDatasetCase1` when `in_channel == 5`, `InpaintDatasetCase2` when `in_channel == 6`
- Facies detection uses exact floating-point comparisons (`img == 1`, `img == -0.0039215684`, etc.)

### New Data Flow (Flumy Pipeline)

```
Flumy simulation → 3D facies block (nx, ny, nz)
                 → Extract plan-view slices at multiple depths
                 → For each slice:
                     Facies → AI → Reflectivity → Ricker convolution → RMS amplitude
                 → Save (facies, RMS) pairs as .npz files
                 → FlumyDataset loads .npz, samples wells on-the-fly
                 → 6-channel conditioning:
                   [rms_amplitude, well_presence, sand, bank, mud, y_t]
                 → in_channel = 6, out_channel = 1
```

**Key differences:**
1. **in_channel changed from 5 to 6** — the extra channel is the RMS amplitude (full-coverage seismic attribute)
2. **No separate mask files** — well sampling is done on-the-fly per epoch (data augmentation)
3. **Data format changed from PNG to .npz** — stores integer facies codes + float32 RMS
4. **Facies encoding simplified** — uses exact integer codes {0: mud, 1: bank, 2: sand} normalized to {-1.0, 0.0, 1.0} instead of gray-value-dependent floats
5. **Dataset selection** — uses `config["dataset_type"] == "flumy"` instead of `in_channel` count

---

## New Files Created

### 1. `diffsim/data/flumy_generator.py` — Flumy Simulation Wrapper

Wraps the `flumy` Python package to run channelized reservoir simulations.

**Key class: `FlumyGenerator`**
- Parameters: `nx, ny, mesh, hmax, ng, isbx, zul, dz`
- `generate(seed)` → runs Flumy, returns 3D facies block (nx, ny, nz)
- `_reclassify_facies(fac_raw)` → maps Flumy's 6 internal facies codes to 3 categories:
  - Codes 0, 1 (background + overbank) → `FACIES_MUD = 0`
  - Codes 2, 3 (levee + crevasse) → `FACIES_BANK = 1`
  - Codes 4, 5 (lag + point bar) → `FACIES_SAND = 2`
- `extract_plan_views()` / `extract_cross_sections()` — slice extraction helpers
- `normalize_facies()` / `denormalize_facies()` — convert between int codes and [-1, 1] floats

**Flumy API usage:**
```python
from flumy import Flumy
flsim = Flumy(nx, ny, mesh, verbose)
success = flsim.launch(seed, hmax, isbx, ng, zul)
fac, grain, age = flsim.getBlock(dz, zb=0, nz=nz)
# fac.shape = (nx, ny, nz), codes 0-5
```

### 2. `diffsim/data/seismic.py` — Seismic Forward Modeling

Computes synthetic RMS amplitude from facies via physically-based seismic forward modeling.

**3D Pipeline (`generate_rms_from_facies_3d`):**
1. Facies → Acoustic Impedance (AI): `{mud: 10000, bank: 8500, sand: 7000}` g/cc·m/s
2. AI → Vertical reflectivity: `R = (AI_below - AI_above) / (AI_below + AI_above)`
3. Reflectivity → Synthetic seismic: convolve each vertical trace with Ricker wavelet
4. Synthetic → RMS amplitude: root-mean-square over depth window centered at target z
5. Apply lateral Gaussian smoothing (simulates Fresnel zone)
6. Add band-limited noise
7. Normalize to [-1, 1]

**2D Fallback (`generate_rms_from_facies_2d`):**
- Simplified: AI → Gaussian smooth → noise → normalize
- Used when 3D block is not available

**Important parameter:** `rms_window_half` must be large enough (~10 samples) to capture facies boundaries within the RMS window. Too small → zero RMS in uniform regions.

### 3. `diffsim/data/well_sampling.py` — Sparse Well Sampling

Simulates exploration/appraisal wells by randomly selecting pixel locations.

- `sample_well_locations(image_size, n_wells, min_spacing, rng)` — random positions with optional minimum spacing constraint (greedy algorithm)
- `create_well_mask(image_size, well_positions)` — binary mask: 0 at wells (known), 1 elsewhere (to generate). Matches DiffSim convention.
- `create_well_conditioning(facies_map, well_positions)` — creates 4-channel conditioning: `[well_presence, sand_at_wells, bank_at_wells, mud_at_wells]`

### 4. `diffsim/data/flumy_dataset.py` — PyTorch Dataset

**Class: `FlumyDataset(torch.utils.data.Dataset)`**

Loads pre-generated `.npz` files and creates conditioning tensors on-the-fly.

- `__init__`: finds all `.npz` files in `data_root`, configures well sampling params
- `__getitem__`: loads facies + RMS, randomly samples well locations (different each epoch = augmentation), builds 5-channel conditioning `[rms, well_presence, sand, bank, mud]`, returns dict matching DiffSim format:
  ```python
  {
      'gt_image':   (1, H, W),    # normalized facies [-1, 1]
      'cond_image': (5, H, W),    # conditioning channels
      'yt_image':   (1, H, W),    # ground truth at wells + noise elsewhere
      'mask_image': (1, H, W),    # for visualization
      'mask':       (1, H, W),    # diffusion mask (0=known, 1=generate)
      'path':       str,
  }
  ```

### 5. `scripts/generate_flumy_data.py` — Data Generation Script

Main entry point for generating training data. Can be run standalone or from config:

```bash
# From command line
python scripts/generate_flumy_data.py \
    --output_dir data/flumy_dataset \
    --n_simulations 100 \
    --slices_per_sim 10 \
    --image_size 64

# From config
python scripts/generate_flumy_data.py --config configs/case1_flumy.json
```

**What it does:**
1. Runs N Flumy simulations with randomized geological parameters (varied net-to-gross and sand body extension)
2. For each simulation, extracts multiple plan-view slices at different depths
3. For each slice, computes RMS amplitude from the full 3D block (physically rigorous)
4. Resizes facies (nearest neighbor) and RMS (bilinear) to target image size
5. Saves as compressed `.npz` with train/test split (by simulation, not by slice)
6. Skips uniform slices (all one facies) and failed simulations

### 6. `configs/case1_flumy.json` — Configuration

New config file for the Flumy pipeline. Key differences from `case1_geomodeling.json`:

| Field | Original | Flumy |
|-------|----------|-------|
| `name` | `case1_geomodeling` | `case1_flumy` |
| `dataset_type` | (not present) | `"flumy"` |
| `in_channel` | 5 | 6 |
| `data.train_image` + `data.train_mask` | separate image/mask dirs | `data.train_data` (single dir with .npz) |
| `data.n_wells_range` | (not present) | `[3, 15]` |
| `data.min_well_spacing` | (not present) | `4` |
| `data_generation` section | (not present) | full Flumy + seismic params |

### 7. `tests/test_flumy_pipeline.py` and `tests/test_flumy_pipeline_standalone.py`

Test files for the pipeline. The standalone version uses `importlib` to load modules directly, bypassing `diffsim/__init__.py` which imports PyTorch at the top level. Tests cover:
- Ricker wavelet generation
- Facies normalization round-trip
- AI mapping
- 3D reflectivity computation
- 3D and 2D RMS generation
- Well sampling with spacing constraints
- Well conditioning channel creation

---

## Modified Files

### `scripts/train_conditional.py`

**Import added:**
```python
from diffsim.data.flumy_dataset import FlumyDataset
```

**Dataset selection logic updated** (around line 155):
```python
# NEW: checks for dataset_type first
dataset_type = config.get("dataset_type", None)
image_size = config["image_size"]

if dataset_type == "flumy":
    # Flumy dataset: loads .npz files with (facies, RMS) pairs
    train_data_path = data_cfg.get("train_data", data_path)
    n_wells_range = tuple(data_cfg.get("n_wells_range", [3, 15]))
    min_well_spacing = data_cfg.get("min_well_spacing", 4)
    dataset = FlumyDataset(
        data_root=train_data_path,
        image_size=(image_size, image_size),
        n_wells_range=n_wells_range,
        min_well_spacing=min_well_spacing,
    )
elif is_3d:
    # ... existing 3D path unchanged
else:
    # ... existing 2D Case1/Case2 path unchanged
```

### `diffsim/data/__init__.py`

Added exports for all new modules: `FlumyGenerator`, `FlumyDataset`, seismic functions, well sampling functions, and facies constants.

### `requirements.txt`

Added `flumy>=8.0` dependency.

---

## Conditioning Channel Comparison

### Original (5 channels = `in_channel` 5)
| Channel | Content |
|---------|---------|
| 0 | Known mask (1 - diffusion_mask) |
| 1 | Sand probability at known locations |
| 2 | Bank probability at known locations |
| 3 | Mud probability at known locations |
| 4 | y_t (noisy starting image) |

### Flumy (6 channels = `in_channel` 6)
| Channel | Content |
|---------|---------|
| 0 | **RMS amplitude** (full coverage, normalized [-1, 1]) |
| 1 | Well presence mask (1 at wells) |
| 2 | Sand indicator at wells |
| 3 | Bank indicator at wells |
| 4 | Mud indicator at wells |
| 5 | y_t (noisy starting image) |

The key architectural change: channel 0 is now **RMS amplitude** (a continuous-valued, spatially-smooth seismic attribute that covers the entire image) rather than a binary known/unknown mask. This gives the network a strong spatial prior for the overall facies distribution.

---

## Known Considerations

1. **Flumy facies reclassification** — The mapping from Flumy's 6 internal codes to our 3 categories may need tuning based on visual inspection of actual Flumy outputs. The current mapping is based on the Flumy documentation.

2. **RMS window size** — The `rms_window_half` parameter (default 10 in config) controls how much vertical context is captured. Too small → zero RMS in uniform layers. Too large → loss of vertical resolution. Should be tuned based on typical Flumy layer thicknesses.

3. **Acoustic impedance values** — The default AI values (sand=7000, bank=8500, mud=10000) are reasonable for shallow clastics but may need calibration for specific geological settings.

4. **No unconditional model** — The Flumy config only defines the conditional model. An unconditional model could be added by training on facies images alone (same as original Case 1 unconditional).

5. **The `diffsim/__init__.py` imports torch at the top level** — This means you cannot import any `diffsim.data.*` module without torch installed. The standalone test works around this with `importlib`. Not a problem for training (torch is always available) but affects standalone data generation if torch isn't installed.
