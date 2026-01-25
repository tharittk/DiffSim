"""
Utility functions for training and data processing.
"""

import random
import numpy as np
import math
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    """
    Convert a torch Tensor into an image Numpy array.

    Args:
        tensor: Input tensor, 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W)
        out_type: Output data type
        min_max: Min and max values for clamping

    Returns:
        Image as numpy array, 3D(H,W,C) or 2D(H,W), [0,255]
    """
    tensor = tensor.clamp_(*min_max)
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = ((img_np + 1) * 127.5).round()
    return img_np.astype(out_type).squeeze()


def postprocess(images):
    """Convert list of tensors to list of numpy images."""
    return [tensor2img(image) for image in images]


def set_seed(seed, gl_seed=0):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
        gl_seed: Global seed offset (used in worker_init_fn)
    """
    if seed >= 0 and gl_seed >= 0:
        seed += gl_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Speed-reproducibility tradeoff
    if seed >= 0 and gl_seed >= 0:  # Slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # Faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def set_gpu(args, distributed=False, rank=0):
    """Set parameter to GPU or DDP."""
    if args is None:
        return None
    if distributed and isinstance(args, torch.nn.Module):
        return DDP(args.cuda(), device_ids=[rank], output_device=rank,
                   broadcast_buffers=True, find_unused_parameters=True)
    else:
        return args.cuda()


def set_device(args, distributed=False, rank=0):
    """Set parameter to GPU or CPU."""
    if torch.cuda.is_available():
        if isinstance(args, list):
            return (set_gpu(item, distributed, rank) for item in args)
        elif isinstance(args, dict):
            return {key: set_gpu(args[key], distributed, rank) for key in args}
        else:
            args = set_gpu(args, distributed, rank)
    return args


def num_to_groups(num, divisor):
    """
    Divide num into groups of size divisor.

    Args:
        num: Total number
        divisor: Group size

    Returns:
        List of group sizes
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
