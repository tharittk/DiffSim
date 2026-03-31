"""Quick test of the Flumy data generation pipeline (no Flumy or PyTorch needed).

Uses importlib to load modules directly, bypassing diffsim/__init__.py
which imports PyTorch models.
"""
import sys
import os
import importlib.util

# Direct module loading to avoid torch imports from diffsim/__init__.py
_data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "diffsim", "data"
)


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load pure-numpy modules without triggering torch
flumy_gen = _load_module(
    "diffsim.data.flumy_generator",
    os.path.join(_data_dir, "flumy_generator.py")
)
seismic = _load_module(
    "diffsim.data.seismic",
    os.path.join(_data_dir, "seismic.py")
)
well_samp = _load_module(
    "diffsim.data.well_sampling",
    os.path.join(_data_dir, "well_sampling.py")
)

import numpy as np


def test_ricker_wavelet():
    print("=== Test 1: Ricker wavelet ===")
    t, w = seismic.ricker_wavelet(25.0, 0.001, 0.1)
    print(f"  Wavelet length: {len(w)}, peak: {w.max():.3f}")
    assert len(w) > 0
    assert abs(w.max() - 1.0) < 0.01  # peak should be ~1


def test_facies_normalization():
    print("=== Test 2: Facies normalization ===")
    facies = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int8)
    norm = flumy_gen.normalize_facies(facies)
    print(f"  Input:  {facies.flatten()}")
    print(f"  Output: {norm.flatten()}")
    denorm = flumy_gen.denormalize_facies(norm)
    print(f"  Roundtrip: {denorm.flatten()}")
    assert np.all(facies == denorm), "Roundtrip failed!"


def test_ai_mapping():
    print("=== Test 3: AI mapping ===")
    facies = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int8)
    ai = seismic.facies_to_ai(facies)
    print(f"  AI values: {ai.flatten()}")
    assert ai[0, 0] == 10000.0  # mud
    assert ai[0, 1] == 8500.0   # bank
    assert ai[0, 2] == 7000.0   # sand


def test_3d_rms_pipeline():
    print("=== Test 4: 3D RMS pipeline ===")
    SAND = flumy_gen.FACIES_SAND
    BANK = flumy_gen.FACIES_BANK
    block = np.zeros((10, 10, 30), dtype=np.int8)
    block[:, :, 10:20] = SAND
    block[:, :, 5:10] = BANK
    ai_block = seismic.facies_to_ai(block)
    refl = seismic.compute_reflectivity_vertical(ai_block)
    print(f"  AI block shape: {ai_block.shape}")
    print(f"  Reflectivity shape: {refl.shape}")
    print(f"  Max |R|: {np.max(np.abs(refl)):.4f}")
    assert refl.shape == (10, 10, 29)
    assert np.max(np.abs(refl)) > 0

    rms = seismic.generate_rms_from_facies_3d(
        block, z_target=15, rms_window_half=3,
        noise_level=0.02, smooth_sigma=0.5
    )
    print(f"  RMS shape: {rms.shape}, range: [{rms.min():.3f}, {rms.max():.3f}]")
    assert rms.shape == (10, 10)
    assert rms.min() >= -1.0 and rms.max() <= 1.0


def test_2d_rms():
    print("=== Test 5: RMS from 2D ===")
    SAND = flumy_gen.FACIES_SAND
    BANK = flumy_gen.FACIES_BANK
    facies_mixed = np.zeros((64, 64), dtype=np.int8)
    facies_mixed[20:40, 15:50] = SAND
    facies_mixed[18:20, 10:55] = BANK
    facies_mixed[40:42, 10:55] = BANK
    rms_mixed = seismic.generate_rms_from_facies_2d(
        facies_mixed, smooth_sigma=3.0, noise_level=0.05
    )
    print(f"  RMS shape: {rms_mixed.shape}")
    print(f"  RMS range: [{rms_mixed.min():.3f}, {rms_mixed.max():.3f}]")
    assert rms_mixed.shape == (64, 64)
    assert rms_mixed.min() >= -1.1 and rms_mixed.max() <= 1.1


def test_well_sampling():
    print("=== Test 6: Well sampling ===")
    SAND = flumy_gen.FACIES_SAND
    BANK = flumy_gen.FACIES_BANK
    facies_mixed = np.zeros((64, 64), dtype=np.int8)
    facies_mixed[20:40, 15:50] = SAND
    facies_mixed[18:20, 10:55] = BANK
    facies_mixed[40:42, 10:55] = BANK

    rng = np.random.default_rng(42)
    wells = well_samp.sample_well_locations(
        (64, 64), n_wells=8, min_spacing=8, rng=rng
    )
    print(f"  Wells: {wells.shape[0]} positions")
    assert wells.shape[0] == 8
    assert wells.shape[1] == 2

    mask = well_samp.create_well_mask((64, 64), wells)
    n_known = int((mask == 0).sum())
    n_unknown = int((mask == 1).sum())
    print(f"  Known pixels: {n_known}, Unknown: {n_unknown}")
    assert n_known == 8
    assert n_unknown == 64 * 64 - 8

    well_cond = well_samp.create_well_conditioning(facies_mixed, wells)
    print(f"  Conditioning shape: {well_cond.shape}")
    assert well_cond.shape == (4, 64, 64)
    for i, name in enumerate(["presence", "sand", "bank", "mud"]):
        print(f"    {name}: {well_cond[i].sum():.0f}")
    # All wells accounted for
    assert well_cond[0].sum() == 8
    assert well_cond[1].sum() + well_cond[2].sum() + well_cond[3].sum() == 8


if __name__ == "__main__":
    test_ricker_wavelet()
    test_facies_normalization()
    test_ai_mapping()
    test_3d_rms_pipeline()
    test_2d_rms()
    test_well_sampling()
    print()
    print("All tests passed!")
