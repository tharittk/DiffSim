"""End-to-end Flumy + RMS + well-conditioning pipeline tests.

This test file intentionally uses the real `flumy` and `torch` stack available
in the environment to validate the current Flumy data pipeline after recent
changes.
"""

from pathlib import Path
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from diffsim.data.flumy_dataset import FlumyDataset
from diffsim.data.flumy_generator import (
    FlumyGenerator,
    denormalize_facies,
    normalize_facies,
)
from diffsim.data.seismic import generate_rms_from_facies_3d


def _generate_small_block():
    # Fail fast with a clear message when the runtime environment is incomplete.
    import flumy  # noqa: F401

    generator = FlumyGenerator(
        nx=24,
        ny=24,
        mesh=10,
        hmax=2.0,
        ng=45,
        isbx=80,
        zul=3.0,
        dz=0.2,
        verbose=False,
    )
    block = generator.generate(seed=7)
    return generator, block


def test_flumy_generate_small_block():
    generator, block = _generate_small_block()
    assert block.shape == (generator.nx, generator.ny, generator.nz)
    assert block.dtype == np.int8

    codes = set(np.unique(block).tolist())
    assert codes.issubset({0, 1, 2})


def test_normalize_roundtrip_from_generated_slice():
    _, block = _generate_small_block()
    facies_2d = block[:, :, block.shape[2] // 2]
    normalized = normalize_facies(facies_2d)
    restored = denormalize_facies(normalized)

    assert normalized.shape == facies_2d.shape
    assert np.all(restored == facies_2d)


def test_dataset_pipeline_sample_has_consistent_channels():
    _, block = _generate_small_block()
    z_target = min(6, block.shape[2] - 1)

    facies_2d = block[:, :, z_target]
    rms_2d = generate_rms_from_facies_3d(
        block,
        z_target=z_target,
        rms_window_half=2,
        noise_level=0.02,
        smooth_sigma=0.7,
        rng=np.random.default_rng(123),
    )

    with tempfile.TemporaryDirectory(prefix="flumy_pipeline_") as tmpdir:
        np.savez(Path(tmpdir) / "sample_0000.npz", facies=facies_2d, rms=rms_2d)

        dataset = FlumyDataset(
            data_root=tmpdir,
            image_size=facies_2d.shape,
            n_wells_range=(6, 6),
            min_well_spacing=4,
            data_len=-1,
            seed=11,
        )

        sample = dataset[0]
        gt_image = sample["gt_image"]
        cond_image = sample["cond_image"]
        yt_image = sample["yt_image"]
        mask = sample["mask"]

        h, w = facies_2d.shape
        assert gt_image.shape == (1, h, w)
        assert cond_image.shape == (5, h, w)
        assert yt_image.shape == (1, h, w)
        assert mask.shape == (1, h, w)

        assert gt_image.dtype == torch.float32
        assert cond_image.dtype == torch.float32
        assert yt_image.dtype == torch.float32
        assert mask.dtype == torch.float32

        # Channel-0 in conditioning is well presence and should match (1 - mask).
        well_presence = cond_image[1]
        expected_well_presence = 1.0 - mask[0]
        assert torch.allclose(well_presence, expected_well_presence)

        # Well-presence count equals the number of facies labels exposed at wells.
        known_count = int(well_presence.sum().item())
        facies_at_wells = cond_image[2:5].sum(dim=0)
        assert known_count == int(facies_at_wells.sum().item())

        # `yt_image` should preserve ground truth at known well locations.
        known = mask[0] == 0
        assert torch.allclose(yt_image[0][known], gt_image[0][known])


if __name__ == "__main__":
    test_flumy_generate_small_block()
    test_normalize_roundtrip_from_generated_slice()
    test_dataset_pipeline_sample_has_consistent_channels()
    print("All standalone Flumy pipeline tests passed.")
