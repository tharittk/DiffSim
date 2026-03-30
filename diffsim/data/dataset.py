"""
Dataset classes for 2D and 3D diffusion models.

Supports different facies types:
- Case 1 (channel facies): 3 facies (sand, bank, mud) -> 4 channel conditioning
- Case 2 (mud drape): 4 facies (sand, sbank, smud, mud) -> 5 channel conditioning
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms

from .mask import (
    bbox2mask,
    brush_stroke_mask,
    get_irregular_mask,
    random_bbox,
    random_cropping_bbox,
)


IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    """Check if a file is an image."""
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    """
    Create list of image paths from directory or file list.

    Args:
        dir: Directory path or path to a text file with image paths

    Returns:
        List of image paths
    """
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str_, encoding="utf-8")]
    else:
        images = []
        assert os.path.isdir(dir), "%s is not a valid directory" % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images


def pil_loader(path):
    """Load image as grayscale PIL Image."""
    return Image.open(path).convert("L")


class ImageDataset(data.Dataset):
    """
    Generic image dataset for 2D models.

    Args:
        data_root: Path to image directory
        image_size: Output image size [H, W]
        data_len: Maximum number of samples (-1 for all)
        loader: Image loading function
        augment_horizontal_flip: Whether to apply random horizontal flips
        convert_image_to: Image mode to convert to (e.g., 'L' for grayscale)
    """

    def __init__(
        self,
        data_root,
        image_size=[64, 64],
        data_len=-1,
        loader=pil_loader,
        augment_horizontal_flip=False,
        convert_image_to="L",
    ):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[: int(data_len)]
        else:
            self.imgs = imgs

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size[0], image_size[1])),
                (
                    transforms.RandomHorizontalFlip()
                    if augment_horizontal_flip
                    else transforms.Lambda(lambda x: x)
                ),
                transforms.ToTensor(),
            ]
        )
        self.loader = loader
        self.image_size = image_size
        self.convert_image_to = convert_image_to

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.convert_image_to and img.mode != self.convert_image_to:
            img = img.convert(self.convert_image_to)
        return self.transform(img)

    def __len__(self):
        return len(self.imgs)


class InpaintDatasetCase1(data.Dataset):
    """
    Dataset for Case 1 (Channel Facies) conditional inpainting.

    Facies values (after normalization to [-1, 1]):
    - Sand: ~1
    - Bank: ~-0.004 (gray value ~127)
    - Mud: ~-1

    Conditioning channels: [known_mask, sand, bank, mud] = 4 channels
    Total input channels: 4 + 1 (y_t) = 5

    Args:
        data_root: Tuple of (image_dir, mask_dir) paths
        mask_config: Dictionary with mask configuration including 'mask_mode'
        data_len: Maximum number of samples (-1 for all)
        image_size: Output image size [H, W]
        loader: Image loading function
    """

    def __init__(
        self,
        data_root,
        mask_config={},
        data_len=-1,
        image_size=[64, 64],
        loader=pil_loader,
    ):
        imgs = make_dataset(data_root[0])
        masks = make_dataset(data_root[1])
        if data_len > 0:
            self.imgs = imgs[: int(data_len)]
            self.masks = masks[: int(data_len)]
        else:
            self.imgs = imgs
            self.masks = masks
        self.tfs = transforms.Compose(
            [
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config.get("mask_mode", "file")
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask(index)

        # Create conditioning with probability maps for 3 facies
        # Use tolerance-based comparison to handle floating-point precision
        init_cond_image = img * (1.0 - mask) + mask * (-0.5)
        sandcond = (torch.abs(init_cond_image - 1.0) < 0.05).float()
        bankcond = (torch.abs(init_cond_image - (-0.004)) < 0.05).float()
        mudcond = (torch.abs(init_cond_image - (-1.0)) < 0.05).float()
        cond_image = torch.cat((1.0 - mask, sandcond, bankcond, mudcond), dim=0)

        yt_image = img * (1.0 - mask) + mask * torch.randn_like(img)
        mask_img = img * (1.0 - mask) + mask

        ret["gt_image"] = img
        ret["cond_image"] = cond_image
        ret["yt_image"] = yt_image
        ret["mask_image"] = mask_img
        ret["mask"] = mask
        ret["path"] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self, index):
        """Generate or load mask based on mask_mode."""
        if self.mask_mode == "bbox":
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == "center":
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h // 4, w // 4, h // 2, w // 2))
        elif self.mask_mode == "irregular":
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == "free_form":
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == "hybrid":
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size)
            mask = regular_mask | irregular_mask
        elif self.mask_mode == "file":
            mask = np.asarray(self.loader(self.masks[index])).astype(np.uint8)
            mask = mask.reshape(self.image_size[0], self.image_size[1], 1)
        else:
            raise NotImplementedError(
                f"Mask mode {self.mask_mode} has not been implemented."
            )
        return torch.from_numpy(mask).permute(2, 0, 1).float()


class InpaintDatasetCase2(data.Dataset):
    """
    Dataset for Case 2 (Mud Drape) conditional inpainting.

    Facies values (after normalization to [-1, 1]):
    - Sand: ~1
    - Sandy bank (sbank): ~0.333
    - Sandy mud (smud): ~-0.333
    - Mud: ~-1

    Conditioning channels: [known_mask, sand, sbank, smud, mud] = 5 channels
    Total input channels: 5 + 1 (y_t) = 6

    Args:
        data_root: Tuple of (image_dir, mask_dir) paths
        mask_config: Dictionary with mask configuration including 'mask_mode'
        data_len: Maximum number of samples (-1 for all)
        image_size: Output image size [H, W]
        loader: Image loading function
    """

    def __init__(
        self,
        data_root,
        mask_config={},
        data_len=-1,
        image_size=[64, 64],
        loader=pil_loader,
    ):
        imgs = make_dataset(data_root[0])
        masks = make_dataset(data_root[1])
        if data_len > 0:
            self.imgs = imgs[: int(data_len)]
            self.masks = masks[: int(data_len)]
        else:
            self.imgs = imgs
            self.masks = masks
        self.tfs = transforms.Compose(
            [
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config.get("mask_mode", "file")
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask(index)

        # Create conditioning with probability maps for 4 facies
        # Use tolerance-based comparison to handle floating-point precision
        init_cond_image = img * (1.0 - mask) + mask * (-0.5)
        sandcond = (torch.abs(init_cond_image - 1.0) < 0.05).float()
        sbankcond = (torch.abs(init_cond_image - 0.333) < 0.05).float()
        smudcond = (torch.abs(init_cond_image - (-0.333)) < 0.05).float()
        mudcond = (torch.abs(init_cond_image - (-1.0)) < 0.05).float()
        cond_image = torch.cat(
            (1.0 - mask, sandcond, sbankcond, smudcond, mudcond), dim=0
        )

        yt_image = img * (1.0 - mask) + mask * torch.randn_like(img)
        mask_img = img * (1.0 - mask) + mask

        ret["gt_image"] = img
        ret["cond_image"] = cond_image
        ret["yt_image"] = yt_image
        ret["mask_image"] = mask_img
        ret["mask"] = mask
        ret["path"] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self, index):
        """Generate or load mask based on mask_mode. Note: file mode inverts the mask."""
        if self.mask_mode == "bbox":
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == "center":
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h // 4, w // 4, h // 2, w // 2))
        elif self.mask_mode == "irregular":
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == "free_form":
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == "hybrid":
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size)
            mask = regular_mask | irregular_mask
        elif self.mask_mode == "file":
            # Note: Case 2 inverts the mask from file
            mask = 1 - np.asarray(self.loader(self.masks[index])).astype(np.uint8)
            mask = mask.reshape(self.image_size[0], self.image_size[1], 1)
        else:
            raise NotImplementedError(
                f"Mask mode {self.mask_mode} has not been implemented."
            )
        return torch.from_numpy(mask).permute(2, 0, 1).float()


# Alias for backwards compatibility
InpaintDataset = InpaintDatasetCase1


class NPYDataset(data.Dataset):
    """
    Dataset for 3D volumetric data stored as .npy files.

    Args:
        folder: Path to folder containing .npy files
        max_files: Maximum number of files to load
        transform: Optional transform to apply
        normalize_max: Maximum value for normalization
    """

    def __init__(self, folder, max_files=9500, transform=None, normalize_max=3):
        super().__init__()
        self.folder = folder
        self.paths = sorted(Path(folder).glob("*.npy"))[:max_files]
        self.transform = transform
        self.normalize_max = normalize_max

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        volume = np.load(path)  # shape: (D, H, W)

        # Normalize and add channel dimension
        volume = torch.tensor(
            volume / self.normalize_max, dtype=torch.float32
        ).unsqueeze(0)

        if self.transform:
            volume = self.transform(volume)

        return volume


class NPYInpaintDataset(data.Dataset):
    """
    Dataset for 3D volumetric inpainting with masks (Case 3: LA3D).

    Similar to InpaintDatasetCase2, with 4 facies types:
    - Sand (lateral accretion): normalized value 1
    - Sandy Bank (channel fill): normalized value 0.333
    - Sandy Mud (mud drapes): normalized value -0.333
    - Mud (floodplain): normalized value -1

    Conditioning channels: [known_mask, sand, sbank, smud, mud] = 5 channels
    Total input channels: 5 + 1 (y_t) = 6

    Args:
        data_folder: Path to folder containing .npy data files (or tuple of (image_dir, mask_dir))
        mask_folder: Path to folder containing .npy mask files (ignored if data_folder is tuple)
        max_files: Maximum number of files to load
        mask_config: Dictionary with mask configuration including 'mask_mode'
        image_size: Volume size [D, H, W]
    """

    def __init__(
        self,
        data_folder,
        mask_folder=None,
        max_files=9500,
        mask_config={},
        image_size=[32, 48, 48],
    ):
        super().__init__()

        # Handle both tuple format and separate arguments
        if isinstance(data_folder, tuple):
            img_dir, mask_dir = data_folder
        else:
            img_dir = data_folder
            mask_dir = mask_folder

        self.data_paths = sorted(Path(img_dir).glob("*.npy"))[:max_files]
        self.mask_paths = sorted(Path(mask_dir).glob("*.npy"))[:max_files]
        self.mask_config = mask_config
        self.mask_mode = mask_config.get("mask_mode", "file")
        self.image_size = image_size

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        ret = {}

        # Load volume (raw values: 0, 1, 2, 3 for 4 facies)
        # Normalize to [-1, 1]: value / 3 * 2 - 1
        max_val = 3
        volume_raw = np.load(self.data_paths[index])
        img = (
            torch.tensor(volume_raw, dtype=torch.float32).unsqueeze(0) / max_val * 2 - 1
        )

        # Load mask
        mask = self.get_mask(index)

        # Create conditioning with probability maps for 4 facies
        # Same as InpaintDatasetCase2
        init_cond_image = img * (1.0 - mask) + mask * (-0.5)
        sandcond = (init_cond_image == 1) + 0.0
        sbankcond = (init_cond_image == 0.33333337) + 0.0
        smudcond = (init_cond_image == -0.3333333) + 0.0
        mudcond = (init_cond_image == -1) + 0.0

        cond_image = np.concatenate(
            (1.0 - mask, sandcond, sbankcond, smudcond, mudcond), axis=0
        )

        yt_image = img * (1.0 - mask) + mask * torch.randn_like(img)
        mask_img = img * (1.0 - mask) + mask

        ret["gt_image"] = img
        ret["cond_image"] = cond_image
        ret["yt_image"] = yt_image
        ret["mask_image"] = mask_img
        ret["mask"] = mask
        ret["path"] = str(self.data_paths[index].name)
        return ret

    def get_mask(self, index):
        """Load mask from file. Mask: 1 = unknown (to inpaint), 0 = known."""
        if self.mask_mode == "file":
            # Load and invert mask (original: 1=known, we need: 1=unknown)
            mask = 1 - np.load(self.mask_paths[index]).astype(np.float32)
            d, h, w = self.image_size
            mask = mask.reshape(d, h, w, 1)
        else:
            raise NotImplementedError(
                f"Mask mode {self.mask_mode} has not been implemented for 3D."
            )
        return torch.from_numpy(mask).permute(3, 0, 1, 2).float()
