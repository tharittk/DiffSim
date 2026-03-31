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

    Args:
        nx: Number of grid nodes along x-axis
        ny: Number of grid nodes along y-axis
        mesh: Horizontal grid mesh size in meters
        hmax: Maximum channel depth in meters
        ng: Target net-to-gross ratio (0-100)
        isbx: Sand body extension parameter (controls cutoff frequency)
        zul: Total reservoir height to fill in meters (default: 3 * hmax)
        dz: Vertical discretization step in meters (default: hmax / 30)
        verbose: Whether to print simulation progress
    """

    def __init__(
        self,
        nx=250,
        ny=250,
        mesh=10,
        hmax=3.0,
        ng=50,
        isbx=80,
        zul=None,
        dz=None,
        verbose=False,
    ):
        self.nx = nx
        self.ny = ny
        self.mesh = mesh
        self.hmax = hmax
        self.ng = ng
        self.isbx = isbx
        self.zul = zul or 3 * hmax
        self.dz = dz or hmax / 30
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

        fac_raw, _, _ = flsim.getBlock(self.dz, zb=0, nz=self.nz)

        # Reclassify flumy facies into 3 categories
        facies_block = self._reclassify_facies(fac_raw)
        return facies_block

    def _reclassify_facies(self, fac_raw):
        """
        Reclassify Flumy's internal facies codes into 3 categories.

        Flumy outputs facies codes that typically include:
            0: Not deposited / Background → mud
            1: Overbank (floodplain mud) → mud
            2: Levee → bank
            3: Crevasse splay → bank
            4: Channel lag → sand
            5: Point bar (channel sand) → sand

        We reclassify into:
            0 (mud): codes 0, 1 (background + overbank)
            1 (bank): codes 2, 3 (levee + crevasse)
            2 (sand): codes 4, 5 (lag + point bar)

        Args:
            fac_raw: Raw flumy facies array

        Returns:
            Reclassified array with values {0, 1, 2}
        """
        facies = np.zeros_like(fac_raw, dtype=np.int8)
        # Mud: background + overbank
        facies[(fac_raw == 0) | (fac_raw == 1)] = FACIES_MUD
        # Bank: levee + crevasse splay
        facies[(fac_raw == 2) | (fac_raw == 3)] = FACIES_BANK
        # Sand: channel lag + point bar
        facies[(fac_raw >= 4)] = FACIES_SAND
        return facies

    def extract_plan_views(self, facies_block, z_indices=None):
        """
        Extract 2D plan-view (horizontal) slices from a 3D facies block.

        Args:
            facies_block: 3D array of shape (nx, ny, nz)
            z_indices: List of z-indices to extract. If None, extracts all.

        Returns:
            List of 2D arrays of shape (nx, ny)
        """
        if z_indices is None:
            z_indices = range(facies_block.shape[2])
        return [facies_block[:, :, z] for z in z_indices]

    def extract_cross_sections(self, facies_block, y_indices=None):
        """
        Extract 2D cross-flow (x-z) sections from a 3D facies block.

        Args:
            facies_block: 3D array of shape (nx, ny, nz)
            y_indices: List of y-indices to extract. If None, extracts all.

        Returns:
            List of 2D arrays of shape (nx, nz)
        """
        if y_indices is None:
            y_indices = range(facies_block.shape[1])
        return [facies_block[:, y, :] for y in y_indices]


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
