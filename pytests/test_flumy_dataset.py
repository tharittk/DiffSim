"""Unit tests for diffsim.data.flumy_dataset.FlumyDataset."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffsim.data.flumy_dataset import FlumyDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _create_dataset_dir(tmp_path, n_samples=3, shape=(16, 16), facies_codes=(0, 1, 2)):
    """Create a minimal facies+rms directory structure with paired .npy files."""
    facies_dir = tmp_path / "facies"
    rms_dir = tmp_path / "rms"

    facies_dir.mkdir()
    rms_dir.mkdir()

    rng = np.random.default_rng(42)
    for i in range(n_samples):
        facies = rng.choice(list(facies_codes), size=shape).astype(np.int8)
        rms = rng.uniform(0.0, 100.0, size=shape).astype(np.float32)
        name = f"sample_{i:03d}.npy"
        np.save(facies_dir / name, facies)
        np.save(rms_dir / name, rms)
    return tmp_path


# ---------------------------------------------------------------------------
# __init__ — directory and pairing validation
# ---------------------------------------------------------------------------
class TestFlumyDatasetInit:
    def test_loads_paired_files(self, tmp_path):
        _create_dataset_dir(tmp_path, n_samples=5)
        ds = FlumyDataset(str(tmp_path), image_size=(16, 16))
        assert len(ds) == 5

    def test_missing_dirs_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Expected directories"):
            FlumyDataset(str(tmp_path), image_size=(16, 16))

    def test_missing_rms_counterpart_raises(self, tmp_path):
        facies_dir = tmp_path / "facies"
        rms_dir = tmp_path / "rms"
        facies_dir.mkdir()
        rms_dir.mkdir()
        np.save(facies_dir / "orphan.npy", np.zeros((8, 8), dtype=np.int8))
        with pytest.raises(FileNotFoundError, match="Missing RMS file"):
            FlumyDataset(str(tmp_path), image_size=(8, 8))

    def test_no_files_raises(self, tmp_path):
        (tmp_path / "facies").mkdir()
        (tmp_path / "rms").mkdir()
        with pytest.raises(FileNotFoundError, match="No paired"):
            FlumyDataset(str(tmp_path), image_size=(8, 8))

    def test_data_len_limits_samples(self, tmp_path):
        _create_dataset_dir(tmp_path, n_samples=10)
        ds = FlumyDataset(str(tmp_path), image_size=(16, 16), data_len=3)
        assert len(ds) == 3

    def test_image_size_as_list(self, tmp_path):
        _create_dataset_dir(tmp_path, n_samples=1)
        ds = FlumyDataset(str(tmp_path), image_size=[16, 16])
        assert ds.image_size == (16, 16)


# ---------------------------------------------------------------------------
# __getitem__ — output structure and types
# ---------------------------------------------------------------------------
class TestFlumyDatasetGetitem:
    @pytest.fixture()
    def dataset(self, tmp_path):
        _create_dataset_dir(tmp_path, n_samples=3, shape=(16, 16))
        return FlumyDataset(str(tmp_path), image_size=(16, 16))

    def test_returns_dict_with_expected_keys(self, dataset):
        sample = dataset[0]
        expected_keys = {
            "gt_image",
            "cond_image",
            "yt_image",
            "mask_image",
            "mask",
            "path",
        }
        assert set(sample.keys()) == expected_keys

    def test_gt_image_shape_and_type(self, dataset):
        sample = dataset[0]
        assert isinstance(sample["gt_image"], torch.Tensor)
        assert sample["gt_image"].shape == (1, 16, 16)
        assert sample["gt_image"].dtype == torch.float32

    def test_cond_image_shape_and_type(self, dataset):
        sample = dataset[0]
        assert isinstance(sample["cond_image"], torch.Tensor)
        assert sample["cond_image"].shape == (1, 16, 16)
        assert sample["cond_image"].dtype == torch.float32

    def test_mask_shape_and_all_ones(self, dataset):
        sample = dataset[0]
        assert sample["mask"].shape == (1, 16, 16)
        assert torch.all(sample["mask"] == 1.0)

    def test_yt_image_shape(self, dataset):
        sample = dataset[0]
        assert sample["yt_image"].shape == (1, 16, 16)

    def test_mask_image_shape(self, dataset):
        sample = dataset[0]
        assert sample["mask_image"].shape == (1, 16, 16)

    def test_path_is_string(self, dataset):
        sample = dataset[0]
        assert isinstance(sample["path"], str)
        assert sample["path"].endswith(".npy")

    def test_gt_image_values_in_normalized_range(self, dataset):
        sample = dataset[0]
        gt = sample["gt_image"]
        # normalize_facies maps {0,1,2} → {-1, 0, 1}
        unique_vals = set(gt.unique().tolist())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_cond_image_normalized_range(self, dataset):
        sample = dataset[0]
        cond = sample["cond_image"]
        assert cond.min() >= -1.0 - 1e-5
        assert cond.max() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# __getitem__ — resizing
# ---------------------------------------------------------------------------
class TestFlumyDatasetResize:
    def test_resize_larger_to_smaller(self, tmp_path):
        _create_dataset_dir(tmp_path, n_samples=1, shape=(32, 32))
        ds = FlumyDataset(str(tmp_path), image_size=(16, 16))
        sample = ds[0]
        assert sample["gt_image"].shape == (1, 16, 16)
        assert sample["cond_image"].shape == (1, 16, 16)

    def test_resize_preserves_facies_codes(self, tmp_path):
        _create_dataset_dir(tmp_path, n_samples=1, shape=(32, 32))
        ds = FlumyDataset(str(tmp_path), image_size=(16, 16))
        gt = ds[0]["gt_image"]
        unique_vals = set(gt.unique().tolist())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_no_resize_when_size_matches(self, tmp_path):
        _create_dataset_dir(tmp_path, n_samples=1, shape=(64, 64))
        ds = FlumyDataset(str(tmp_path), image_size=(64, 64))
        sample = ds[0]
        assert sample["gt_image"].shape == (1, 64, 64)


# ---------------------------------------------------------------------------
# Invalid facies codes
# ---------------------------------------------------------------------------
class TestInvalidFaciesCodes:
    def test_bad_facies_code_raises(self, tmp_path):
        facies_dir = tmp_path / "facies"
        rms_dir = tmp_path / "rms"
        facies_dir.mkdir()
        rms_dir.mkdir()
        # Facies with invalid code 5
        bad_facies = np.full((8, 8), 5, dtype=np.int8)
        rms = np.ones((8, 8), dtype=np.float32)
        np.save(facies_dir / "bad.npy", bad_facies)
        np.save(rms_dir / "bad.npy", rms)
        ds = FlumyDataset(str(tmp_path), image_size=(8, 8))
        with pytest.raises(ValueError, match="unexpected facies codes"):
            ds[0]


# ---------------------------------------------------------------------------
# DataLoader integration (torch)
# ---------------------------------------------------------------------------
class TestDataLoaderIntegration:
    @pytest.fixture()
    def dataset(self, tmp_path):
        _create_dataset_dir(tmp_path, n_samples=4, shape=(16, 16))
        return FlumyDataset(str(tmp_path), image_size=(16, 16))

    def test_dataloader_batch(self, dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        assert batch["gt_image"].shape == (2, 1, 16, 16)
        assert batch["cond_image"].shape == (2, 1, 16, 16)
        assert batch["mask"].shape == (2, 1, 16, 16)
        assert batch["yt_image"].shape == (2, 1, 16, 16)

    def test_dataloader_iterates_all(self, dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        total = sum(b["gt_image"].shape[0] for b in loader)
        assert total == len(dataset)

    def test_dataloader_batch_dtypes(self, dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        batch = next(iter(loader))
        assert batch["gt_image"].dtype == torch.float32
        assert batch["cond_image"].dtype == torch.float32
        assert batch["mask"].dtype == torch.float32

    def test_cond_image_can_concat_with_noise(self, dataset):
        """Simulate what the training loop does: cat(cond_image, y_t) → 2 channels."""
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        batch = next(iter(loader))
        y_t = torch.randn_like(batch["gt_image"])
        model_input = torch.cat([batch["cond_image"], y_t], dim=1)
        assert model_input.shape == (2, 2, 16, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
