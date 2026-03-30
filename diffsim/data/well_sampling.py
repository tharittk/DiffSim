"""
Sparse well sampling utilities for generating conditioning data.

Simulates well data by randomly selecting column positions from a facies map
and extracting facies values at those locations. Creates binary masks and
per-facies probability maps matching the conditioning format of DiffSim.
"""

import numpy as np

from .flumy_generator import FACIES_SAND, FACIES_BANK, FACIES_MUD


def sample_well_locations(image_size, n_wells, min_spacing=None, rng=None):
    """
    Randomly select well locations on a 2D grid.

    Args:
        image_size: Tuple (height, width) of the facies map
        n_wells: Number of wells to sample
        min_spacing: Minimum spacing between wells in pixels. If None,
                     no constraint.
        rng: numpy random Generator for reproducibility

    Returns:
        well_positions: Array of shape (n_wells, 2) with (row, col) positions
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = image_size

    if min_spacing is None:
        rows = rng.integers(0, h, size=n_wells)
        cols = rng.integers(0, w, size=n_wells)
        return np.stack([rows, cols], axis=1)

    # Sample with minimum spacing constraint (greedy)
    positions = []
    max_attempts = n_wells * 100
    attempts = 0
    while len(positions) < n_wells and attempts < max_attempts:
        r = int(rng.integers(0, h))
        c = int(rng.integers(0, w))
        if all(
            np.sqrt((r - pr) ** 2 + (c - pc) ** 2) >= min_spacing
            for pr, pc in positions
        ):
            positions.append((r, c))
        attempts += 1

    if len(positions) == 0:
        # Fallback: place at least one well at center
        positions.append((h // 2, w // 2))
    elif len(positions) < n_wells:
        import warnings

        warnings.warn(
            f"Could only place {len(positions)}/{n_wells} wells with "
            f"min_spacing={min_spacing} on {h}x{w} grid. "
            f"Consider reducing min_spacing or n_wells.",
            stacklevel=2,
        )

    return np.array(positions, dtype=np.int64)


def create_well_mask(image_size, well_positions):
    """
    Create a binary mask for diffusion: 0 at well locations (known), 1 elsewhere
    (to generate).

    This matches the DiffSim convention where mask=1 means "generate" and
    mask=0 means "keep/known".

    Args:
        image_size: Tuple (height, width)
        well_positions: Array of shape (n_wells, 2) with (row, col) positions

    Returns:
        mask: Float32 array (height, width), 0 at wells, 1 elsewhere
    """
    mask = np.ones(image_size, dtype=np.float32)
    for r, c in well_positions:
        mask[r, c] = 0.0
    return mask


def create_well_conditioning(facies_map, well_positions):
    """
    Create well conditioning channels matching DiffSim inpainting format.

    Produces per-facies binary indicator maps at well locations:
        Channel 0: Well presence mask (1 at wells, 0 elsewhere) = 1 - diffusion_mask
        Channel 1: Sand indicator at wells
        Channel 2: Bank indicator at wells
        Channel 3: Mud indicator at wells

    Args:
        facies_map: 2D array (H, W) with facies codes {0: mud, 1: bank, 2: sand}
        well_positions: Array of shape (n_wells, 2) with (row, col) positions

    Returns:
        well_cond: Array of shape (4, H, W) with conditioning channels
    """
    h, w = facies_map.shape

    well_presence = np.zeros((h, w), dtype=np.float32)
    sand_at_wells = np.zeros((h, w), dtype=np.float32)
    bank_at_wells = np.zeros((h, w), dtype=np.float32)
    mud_at_wells = np.zeros((h, w), dtype=np.float32)

    for r, c in well_positions:
        well_presence[r, c] = 1.0
        fac = facies_map[r, c]
        if fac == FACIES_SAND:
            sand_at_wells[r, c] = 1.0
        elif fac == FACIES_BANK:
            bank_at_wells[r, c] = 1.0
        elif fac == FACIES_MUD:
            mud_at_wells[r, c] = 1.0

    # Stack: [well_presence, sand, bank, mud]
    well_cond = np.stack(
        [well_presence, sand_at_wells, bank_at_wells, mud_at_wells], axis=0
    )
    return well_cond
