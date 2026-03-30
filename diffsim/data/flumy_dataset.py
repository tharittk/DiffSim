"""
Dataset class for Flumy-generated training data with RMS + well conditioning.

Data format:
    Each sample is stored as a .npz file containing:
        - 'facies': 2D facies map (H, W) with codes {0: mud, 1: bank, 2: sand}
        - 'rms': 2D RMS amplitude map (H, W) normalized to [-1, 1]

    Well sampling is done on-the-fly during training for data augmentation.

Conditioning channels (total in_channel = 6):
    Channel 0: RMS amplitude (full coverage, normalized to [-1, 1])
    Channel 1: Well presence mask (1 at well locations, 0 elsewhere)
    Channel 2: Sand indicator at wells
    Channel 3: Bank indicator at wells
    Channel 4: Mud indicator at wells
    Channel 5: y_t (noisy image, concatenated by training loop)
"""

import numpy as np
from pathlib import Path

import torch
import torch.utils.data as data

from .flumy_generator import normalize_facies
from .well_sampling import (
    sample_well_locations,
    create_well_mask,
    create_well_conditioning,
)


class FlumyDataset(data.Dataset):
    """
    Dataset for conditional diffusion training with Flumy-generated data.

    Loads pre-generated (facies, RMS) pairs and creates conditioning tensors
    on-the-fly with random well sampling for augmentation. Each call to
    __getitem__ samples new well locations, providing diverse conditioning
    during training.

    Conditioning structure (5 channels):
        [rms, well_presence, sand_at_wells, bank_at_wells, mud_at_wells]
    Plus 1 noisy channel (y_t) = 6 total input channels
    Output: 1 channel (facies to generate)

    Masking convention (matches DiffSim):
        mask = 0 at well locations (known, preserve during diffusion)
        mask = 1 elsewhere (unknown, generate)

    Args:
        data_root: Path to directory containing .npz files
        image_size: Target image size [H, W] for resizing
        n_wells_range: Tuple (min_wells, max_wells) for random well count
        min_well_spacing: Minimum pixel spacing between wells (None = no limit)
        data_len: Maximum number of samples (-1 for all)
        seed: Random seed for well sampling (None for random)
    """

    def __init__(
        self,
        data_root,
        image_size=(64, 64),
        n_wells_range=(3, 15),
        min_well_spacing=4,
        data_len=-1,
        seed=None,
    ):
        self.data_root = Path(data_root)
        self.image_size = (
            tuple(image_size) if not isinstance(image_size, tuple) else image_size
        )
        self.n_wells_range = n_wells_range
        self.min_well_spacing = min_well_spacing
        self.rng = np.random.default_rng(seed)

        # Find all .npz files
        self.files = sorted(self.data_root.glob("*.npz"))
        if data_len > 0:
            self.files = self.files[:data_len]

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No .npz files found in {data_root}. "
                "Run scripts/generate_flumy_data.py first."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Load pre-generated data
        data_dict = np.load(self.files[index])

        # Validate expected keys
        if "facies" not in data_dict or "rms" not in data_dict:
            raise KeyError(
                f"File {self.files[index]} missing required keys. "
                f"Expected 'facies' and 'rms', got {list(data_dict.keys())}"
            )

        facies_int = data_dict["facies"]  # (H, W) int codes {0, 1, 2}
        rms = data_dict["rms"]  # (H, W) float [-1, 1]

        # Validate facies codes
        unique_codes = set(np.unique(facies_int))
        valid_codes = {0, 1, 2}
        if not unique_codes.issubset(valid_codes):
            raise ValueError(
                f"File {self.files[index]} has unexpected facies codes: "
                f"{unique_codes - valid_codes}. Expected subset of {valid_codes}."
            )

        # Resize if needed
        if facies_int.shape != self.image_size:
            facies_int = self._resize_facies(facies_int, self.image_size)
            rms = self._resize_continuous(rms, self.image_size)

        # Normalize facies to [-1, 1]
        facies_norm = normalize_facies(facies_int)

        # Random well sampling (on-the-fly augmentation)
        n_wells = int(
            self.rng.integers(self.n_wells_range[0], self.n_wells_range[1] + 1)
        )
        well_positions = sample_well_locations(
            self.image_size, n_wells, min_spacing=self.min_well_spacing, rng=self.rng
        )

        # Diffusion mask: 0 at wells (known), 1 elsewhere (generate)
        mask = create_well_mask(self.image_size, well_positions)

        # Well conditioning channels: [well_presence, sand, bank, mud]
        well_cond = create_well_conditioning(facies_int, well_positions)

        # Build full conditioning: [rms, well_presence, sand, bank, mud]
        rms_channel = rms[np.newaxis, :, :]  # (1, H, W)
        cond_image = np.concatenate([rms_channel, well_cond], axis=0)  # (5, H, W)

        # Convert to tensors
        gt_image = torch.from_numpy(facies_norm[np.newaxis, :, :]).float()
        cond_image = torch.from_numpy(cond_image).float()
        mask = torch.from_numpy(mask[np.newaxis, :, :]).float()

        # y_t: ground truth at wells, noise elsewhere
        yt_image = gt_image * (1.0 - mask) + mask * torch.randn_like(gt_image)

        # mask_image: ground truth with mask overlay (for visualization)
        mask_img = gt_image * (1.0 - mask) + mask

        ret = {
            "gt_image": gt_image,  # (1, H, W) normalized facies
            "cond_image": cond_image,  # (5, H, W) conditioning channels
            "yt_image": yt_image,  # (1, H, W) known + noise
            "mask_image": mask_img,  # (1, H, W) for visualization
            "mask": mask,  # (1, H, W) diffusion mask
            "path": self.files[index].name,
        }
        return ret

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
