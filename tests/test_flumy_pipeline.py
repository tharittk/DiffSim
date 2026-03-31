"""Quick test of the Flumy data generation pipeline (no Flumy or PyTorch needed)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Import data modules directly (avoid top-level torch imports in diffsim/__init__.py)
from diffsim.data.seismic import (
    ricker_wavelet,
    facies_to_ai,
    compute_reflectivity_vertical,
    generate_rms_from_facies_3d,
    generate_rms_from_facies_2d,
)
from diffsim.data.well_sampling import (
    sample_well_locations,
    create_well_mask,
    create_well_conditioning,
)
from diffsim.data.flumy_generator import (
    normalize_facies,
    denormalize_facies,
    FACIES_MUD,
    FACIES_BANK,
    FACIES_SAND,
)

print("=== Test 1: Ricker wavelet ===")
t, w = ricker_wavelet(25.0, 0.001, 0.1)
print(f"  Wavelet length: {len(w)}, peak: {w.max():.3f}")

print("=== Test 2: Facies normalization ===")
facies = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int8)
norm = normalize_facies(facies)
print(f"  Input:  {facies.flatten()}")
print(f"  Output: {norm.flatten()}")
denorm = denormalize_facies(norm)
print(f"  Roundtrip: {denorm.flatten()}")
assert np.all(facies == denorm), "Roundtrip failed!"

print("=== Test 3: AI mapping ===")
ai = facies_to_ai(facies)
print(f"  AI values: {ai.flatten()}")

print("=== Test 4: 3D RMS pipeline ===")
block = np.zeros((10, 10, 30), dtype=np.int8)
block[:, :, 10:20] = FACIES_SAND
block[:, :, 5:10] = FACIES_BANK
ai_block = facies_to_ai(block)
refl = compute_reflectivity_vertical(ai_block)
print(f"  AI block shape: {ai_block.shape}")
print(f"  Reflectivity shape: {refl.shape}")
print(f"  Max |R|: {np.max(np.abs(refl)):.4f}")
rms = generate_rms_from_facies_3d(
    block, z_target=15, rms_window_half=3, noise_level=0.02, smooth_sigma=0.5
)
print(f"  RMS shape: {rms.shape}, range: [{rms.min():.3f}, {rms.max():.3f}]")

print("=== Test 5: RMS from 2D ===")
facies_mixed = np.zeros((64, 64), dtype=np.int8)
facies_mixed[20:40, 15:50] = FACIES_SAND
facies_mixed[18:20, 10:55] = FACIES_BANK
facies_mixed[40:42, 10:55] = FACIES_BANK
rms_mixed = generate_rms_from_facies_2d(
    facies_mixed, smooth_sigma=3.0, noise_level=0.05
)
print(
    f"  RMS shape: {rms_mixed.shape}, range: [{rms_mixed.min():.3f}, {rms_mixed.max():.3f}]"
)

print("=== Test 6: Well sampling ===")
rng = np.random.default_rng(42)
wells = sample_well_locations((64, 64), n_wells=8, min_spacing=8, rng=rng)
print(f"  Wells: {wells.shape[0]} positions")
mask = create_well_mask((64, 64), wells)
print(f"  Known pixels: {int((mask == 0).sum())}, Unknown: {int((mask == 1).sum())}")
well_cond = create_well_conditioning(facies_mixed, wells)
print(f"  Conditioning shape: {well_cond.shape}")
for i, name in enumerate(["presence", "sand", "bank", "mud"]):
    print(f"    {name}: {well_cond[i].sum():.0f}")

print()
print("All tests passed!")
