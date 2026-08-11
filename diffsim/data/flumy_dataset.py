"""
Dataset class for Flumy-generated training data with RMS-only conditioning.

Data format:
    Each sample is represented by two files with the same basename:
        - facies/<name>.npy: facies map/cube with codes {0: bank, 1: channel, 2: point_bar}
        - rms/<name>.npy: RMS attribute map/cube (continuous values)

The loader supports both 2D arrays (H, W) and 3D cubes (H, W, Z).
For 3D input, a random aligned depth slice is selected on each __getitem__ call.

Conditioning channels (total in_channel = 2):
    Channel 0: RMS amplitude (full coverage, normalized to [-1, 1])
    Channel 1: y_t (noisy image, concatenated by training loop)
"""

from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from diffsim.data.flumy_generator import normalize_lithofacies
from diffsim.data.seismic import normalize_rms


class FlumyDataset(data.Dataset):
    """
    Dataset for conditional diffusion training with Flumy-generated data.

    Loads pre-generated facies/RMS file pairs and creates RMS-only
    conditioning tensors.

    Conditioning structure (1 channel):
        [rms]
    Plus 1 noisy channel (y_t) = 2 total UNet input channels
    Output: 1 channel (facies to generate)

    Masking convention (matches DiffSim):
        mask = 1 everywhere (fully generative, no sparse-well constraints)

    Args:
        data_root: Path to directory containing facies/ and rms/ folders
        image_size: Target image size [H, W] for resizing
        data_len: Maximum number of samples (-1 for all)
        seed: Random seed for slice selection (None for random)
        facies_subdir: Name of facies subdirectory under data_root
        rms_subdir: Name of RMS subdirectory under data_root
    """

    def __init__(
        self,
        data_root,
        image_size=(64, 64),
        data_len=-1,
        seed=None,
        facies_subdir="facies",
        rms_subdir="rms",
    ):
        self.data_root = Path(data_root)
        self.image_size = (
            tuple(image_size) if not isinstance(image_size, tuple) else image_size
        )
        self.rng = np.random.default_rng(seed)
        self.facies_dir = self.data_root / facies_subdir
        self.rms_dir = self.data_root / rms_subdir

        if not self.facies_dir.is_dir() or not self.rms_dir.is_dir():
            raise FileNotFoundError(
                f"Expected directories {self.facies_dir} and {self.rms_dir}. "
                "Run scripts/generate_flumy_data.py to create split files."
            )

        facies_files = sorted(self.facies_dir.glob("*.npy"))
        self.file_pairs = []
        for facies_file in facies_files:
            rms_file = self.rms_dir / facies_file.name
            if rms_file.is_file():
                self.file_pairs.append((facies_file, rms_file))
            else:
                # Probably too restrictive but a good discipline
                raise FileNotFoundError(
                    f"Missing RMS file {rms_file} for facies file {facies_file}."
                )

        if data_len > 0:
            self.file_pairs = self.file_pairs[:data_len]

        if len(self.file_pairs) == 0:
            missing = [
                p.name for p in facies_files if not (self.rms_dir / p.name).is_file()
            ]
            missing_preview = ", ".join(missing[:5]) if missing else "none"
            raise FileNotFoundError(
                f"No paired facies/RMS .npy files found in {self.data_root}. "
                f"Missing RMS counterparts example: {missing_preview}"
            )

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, index):
        facies_path, rms_path = self.file_pairs[index]
        facies_arr = np.load(facies_path)
        rms_arr = np.load(rms_path)

        facies_int, rms = self._validate_facies_rms(facies_arr, rms_arr)
        facies_int = facies_int.astype(np.int8, copy=False)
        rms = rms.astype(np.float32, copy=False)

        # Validate facies codes for LITHO_FLUMY_AV: 0=Shale, 1=Sand, 2=Silt
        unique_codes = set(np.unique(facies_int))
        valid_codes = {0, 1, 2}
        if not unique_codes.issubset(valid_codes):
            raise ValueError(
                f"File {facies_path} has unexpected facies codes: "
                f"{unique_codes - valid_codes}. Expected subset of {valid_codes}."
            )

        # Resize if needed
        if facies_int.shape != self.image_size:
            facies_int = self._resize_facies(facies_int, self.image_size)
            rms = self._resize_continuous(rms, self.image_size)

        # Normalize channels to the range expected by diffusion training.
        facies_norm = normalize_lithofacies(facies_int)
        rms_norm = normalize_rms(rms, vmin=-1.0, vmax=1.0)

        # RMS-only conditioning with full-generation mask.
        mask = np.ones(self.image_size, dtype=np.float32)
        cond_image = rms_norm[np.newaxis, :, :].astype(np.float32)

        # Convert to tensors
        gt_image = torch.from_numpy(facies_norm[np.newaxis, :, :]).float()
        cond_image = torch.from_numpy(cond_image).float()
        mask = torch.from_numpy(mask[np.newaxis, :, :]).float()

        # y_t: pure noise in RMS-only mode (no known well pixels).
        yt_image = gt_image * (1.0 - mask) + mask * torch.randn_like(gt_image)

        # mask_image: ground truth with mask overlay (for visualization)
        mask_img = gt_image * (1.0 - mask) + mask

        ret = {
            "gt_image": gt_image,  # (1, H, W) normalized facies
            "cond_image": cond_image,  # (1, H, W) RMS conditioning
            "yt_image": yt_image,  # (1, H, W) noise initialization
            "mask_image": mask_img,  # (1, H, W) for visualization
            "mask": mask,  # (1, H, W) diffusion mask
            "path": facies_path.name,
        }
        return ret

    def _validate_facies_rms(self, facies_arr, rms_arr):
        """Return aligned 2D facies/RMS maps. Only 2D arrays are accepted."""
        if facies_arr.ndim != 2:
            raise ValueError(
                f"Expected 2D facies array, got {facies_arr.ndim}D "
                f"(shape {facies_arr.shape}). Save 2D slices instead of 3D cubes."
            )
        if rms_arr.ndim != 2:
            raise ValueError(
                f"Expected 2D RMS array, got {rms_arr.ndim}D "
                f"(shape {rms_arr.shape}). Save 2D slices instead of 3D cubes."
            )
        return facies_arr, rms_arr

    @staticmethod
    def _resize_facies(facies, target_size):
        """Resize discrete facies map using nearest neighbor interpolation."""
        from PIL import Image

        img = Image.fromarray(facies.astype(np.uint8))
        img = img.resize((target_size[1], target_size[0]), Image.NEAREST)
        return np.array(img).astype(np.int8)

    @staticmethod
    def _resize_continuous(arr, target_size):
        """Resize continuous array using bilinear interpolation."""
        from PIL import Image

        img = Image.fromarray(arr.astype(np.float32), mode="F")
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
        return np.array(img, dtype=np.float32)
