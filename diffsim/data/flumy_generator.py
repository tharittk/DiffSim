"""
Flumy simulation wrapper for generating channelized reservoir facies models.

Wraps the flumy Python package (https://flumy.minesparis.psl.eu/) to produce
3D facies blocks and extract 2D plan-view slices for training data generation.

Facies classification (3 types, matching Case 1 convention):
    - Sand (channel fill): code 2, normalized value ~1.0
    - Bank (levee): code 1, normalized value ~0.0
    - Mud (overbank): code 0, normalized value ~-1.0
"""

import numpy as np

# Simplified 3-facies encoding
FACIES_MUD = 0
FACIES_BANK = 1
FACIES_SAND = 2

FACIES_NAMES = {FACIES_MUD: "mud", FACIES_BANK: "bank", FACIES_SAND: "sand"}

# Normalized values matching Case 1 convention [-1, 1]
FACIES_NORMALIZED = {
    FACIES_MUD: -1.0,
    FACIES_BANK: 0.0,
    FACIES_SAND: 1.0,
}


class FlumyGenerator:
    """
    Generate channelized reservoir facies models using the flumy package.

    Produces 3D facies blocks and extracts 2D plan-view slices for training
    data generation. Each simulation generates a unique realization using
    a different random seed.

    NOTE: it is better to fix nx, ny to some multiple of 64 (e.g. 256) to ensure compatibility with common CNN architectures.
    Varying the mesh size according to channel width introduces complexity in the training pipeline. Some channels geting cropped
    out should not be a problem as long as the training data is sufficiently diverse.

    Args:
        nx: Number of grid nodes along x-axis
        ny: Number of grid nodes along y-axis
        mesh: Horizontal grid mesh size in meters
        hmax: Maximum channel depth in meters
        ng: Target net-to-gross ratio (0-100)
        isbx: Sand body extension parameter (controls cutoff frequency)
        zul: Total reservoir height to fill in meters (default: 3 * hmax)
        dz: Vertical discretization step in meters (default: 1)
        verbose: Whether to print simulation progress
    """

    def __init__(
        self,
        nx=256,
        ny=256,
        mesh=20,
        hmax=3.0,
        ng=50,
        isbx=80,
        zul=None,
        dz=1,
        verbose=False,
    ):
        self.nx = nx
        self.ny = ny
        self.mesh = mesh
        self.hmax = hmax
        self.ng = ng
        self.isbx = isbx
        self.zul = zul or 3 * hmax
        self.dz = dz
        self.nz = int(self.zul / self.dz)
        self.verbose = verbose

    def generate(self, seed):
        """
        Run a single Flumy simulation and return the 3D facies block.

        Args:
            seed: Random seed for reproducibility

        Returns:
            facies_block: np.ndarray of shape (nx, ny, nz) with facies codes
                         {0: mud, 1: bank, 2: sand}
        """
        from flumy import Flumy

        flsim = Flumy(self.nx, self.ny, self.mesh, self.verbose)
        success = flsim.launch(seed, self.hmax, self.isbx, self.ng, self.zul)
        if not success:
            raise RuntimeError(f"Flumy simulation failed with seed={seed}")

        facies_raw, _, _ = flsim.getBlock(self.dz, zb=0, nz=self.nz)

        return facies_raw.astype(np.int8)

    @staticmethod
    def reclassify_to_three_facies(facies_raw):
        """
        Ref: User's guide page 121: use Nb groups = 3
        Reclassify Flumy's internal facies codes into 3 categories.

        See User's guide page 120-121 for information.
        """
        # Background mud + overbank
        facies = np.ones(facies_raw.shape, dtype=np.int8) * FACIES_MUD

        # Sand: channel lag + point bar
        facies[
            (facies_raw == 1)
            | (facies_raw == 2)
            | (facies_raw == 3)
            | (facies_raw == 4)
        ] = FACIES_SAND
        # Bank: levee + crevasse splay
        facies[(facies_raw == 5) | (facies_raw == 6) | (facies_raw == 7)] = FACIES_BANK

        return facies


def normalize_facies(facies_map):
    """
    Normalize integer facies codes to [-1, 1] range.

    Args:
        facies_map: 2D array with values {0: mud, 1: bank, 2: sand}

    Returns:
        Normalized array with values {-1.0: mud, 0.0: bank, 1.0: sand}
    """
    normalized = np.zeros_like(facies_map, dtype=np.float32)
    for code, value in FACIES_NORMALIZED.items():
        normalized[facies_map == code] = value
    return normalized


def denormalize_facies(normalized_map):
    """
    Convert normalized facies values back to integer codes.

    Args:
        normalized_map: Array with continuous values, will be rounded to nearest facies

    Returns:
        Integer array with values {0, 1, 2}
    """
    values = np.array(list(FACIES_NORMALIZED.values()))
    codes = np.array(list(FACIES_NORMALIZED.keys()))

    distances = np.abs(
        normalized_map[..., np.newaxis] - values[np.newaxis, np.newaxis, :]
    )
    closest = np.argmin(distances, axis=-1)
    return codes[closest].astype(np.int8)
