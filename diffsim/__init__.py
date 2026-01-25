"""
DiffSim: Diffusion Models for Geomodelling

A library for unconditional and conditional diffusion models
for geological facies modeling, supporting both 2D and 3D data.

Example usage:
    from diffsim import Unet, Unet3D, Network
    from diffsim.core.diffusion import sample, linear_beta_schedule
"""

from .models.unet import Unet
from .models.unet3d import Unet3D
from .core.diffusion import (
    Diffusion,
    sample,
    linear_beta_schedule,
    cosine_beta_schedule
)
from .core.network import Network

__version__ = "0.1.0"

__all__ = [
    'Unet',
    'Unet3D',
    'Diffusion',
    'sample',
    'linear_beta_schedule',
    'cosine_beta_schedule',
    'Network',
]
