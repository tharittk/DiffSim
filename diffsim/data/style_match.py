"""
Amplitude style matching: transfer the statistical distribution of a real
seismic RMS map onto synthetic (Flumy-derived) RMS samples.

The real reference comes from the h05_sub1 scattered-data file:
    col 0  X  (UTM easting)
    col 1  Y  (UTM northing)
    col 2  Z  (RMS amplitude value)
    col 3  column  (integer grid index)
    col 4  row     (integer grid index)

Grid cell spacing is ~25 m (rotated survey grid); Flumy mesh is 20 m.
The spatial mismatch is small enough to ignore for amplitude distribution
matching purposes.

Public API
----------
load_seismic_rms_grid(path)
    Read h05_sub1 → 2-D float32 array (rows × cols), NaN for missing cells.

sample_reference_patches(grid, patch_size, n_patches, rng)
    Draw n_patches valid (no NaN) patches from the grid.

build_reference_cdf(grid)
    Compute sorted value array and CDF from all non-NaN grid values.

histogram_match_rms(source, ref_sorted_vals)
    Map source values through the reference CDF (rank-based transfer).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_seismic_rms_grid(path: str | Path) -> np.ndarray:
    """Read h05_sub1 and return a 2-D RMS grid indexed by (row, col).

    Missing cells (not present in the file) are filled with NaN.

    Returns
    -------
    grid : np.ndarray, shape (n_rows, n_cols), dtype float32
    """
    path = Path(path)
    data = np.loadtxt(path, comments="#", dtype=np.float64)

    z_vals = data[:, 2].astype(np.float32)
    cols = data[:, 3].astype(np.int32)
    rows = data[:, 4].astype(np.int32)

    col_min, col_max = int(cols.min()), int(cols.max())
    row_min, row_max = int(rows.min()), int(rows.max())

    n_rows = row_max - row_min + 1
    n_cols = col_max - col_min + 1

    grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    grid[rows - row_min, cols - col_min] = z_vals
    return grid


# ---------------------------------------------------------------------------
# Reference CDF
# ---------------------------------------------------------------------------

def build_reference_cdf(
    grid: np.ndarray,
    clip_percentile: float = 99.5,
) -> np.ndarray:
    """Return sorted non-NaN values from the grid (used as empirical CDF).

    Parameters
    ----------
    grid : 2-D float32 array with possible NaN cells.
    clip_percentile : upper percentile to clip to before building the CDF.
        Removes acquisition noise spikes.  Set to 100.0 to disable.

    Returns
    -------
    ref_sorted : 1-D float32 array of sorted reference values.
    """
    valid = grid[np.isfinite(grid)]
    if valid.size == 0:
        raise ValueError("Reference grid contains no finite values.")
    if clip_percentile < 100.0:
        upper = float(np.percentile(valid, clip_percentile))
        valid = np.clip(valid, None, upper)
    return np.sort(valid)


# ---------------------------------------------------------------------------
# Patch sampling
# ---------------------------------------------------------------------------

def sample_reference_patches(
    grid: np.ndarray,
    patch_size: int,
    n_patches: int,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Sample random patches from the reference grid, skipping NaN-heavy ones.

    Parameters
    ----------
    grid : 2-D float32 array (n_rows × n_cols).
    patch_size : side length of square patches.
    n_patches : number of patches to return.
    rng : numpy Generator; if None, a fresh one is created.

    Returns
    -------
    List of float32 arrays, each of shape (patch_size, patch_size).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_rows, n_cols = grid.shape
    if n_rows < patch_size or n_cols < patch_size:
        raise ValueError(
            f"Reference grid ({n_rows}×{n_cols}) smaller than patch_size ({patch_size})."
        )

    patches = []
    max_attempts = n_patches * 20
    attempts = 0
    while len(patches) < n_patches and attempts < max_attempts:
        r = rng.integers(0, n_rows - patch_size + 1)
        c = rng.integers(0, n_cols - patch_size + 1)
        patch = grid[r : r + patch_size, c : c + patch_size]
        nan_frac = np.sum(~np.isfinite(patch)) / patch.size
        if nan_frac < 0.05:  # accept patches with <5% missing cells
            patches.append(patch.copy())
        attempts += 1

    if len(patches) < n_patches:
        warnings.warn(
            f"Only {len(patches)}/{n_patches} valid patches found after {max_attempts} attempts."
        )
    return patches


# ---------------------------------------------------------------------------
# Histogram / CDF matching
# ---------------------------------------------------------------------------

def histogram_match_rms(
    source: np.ndarray,
    ref_sorted_vals: np.ndarray,
) -> np.ndarray:
    """Transfer the amplitude distribution of `ref_sorted_vals` onto `source`.

    Uses rank-based (quantile) matching — identical to
    ``skimage.exposure.match_histograms`` but without the skimage dependency.

    Parameters
    ----------
    source : float32 array of any shape (the Flumy-derived RMS patch/crop).
    ref_sorted_vals : 1-D sorted float32 array from ``build_reference_cdf``.

    Returns
    -------
    matched : float32 array, same shape as source.
    """
    src_flat = source.ravel().astype(np.float64)
    # Rank each source value within [0, 1]
    order = np.argsort(src_flat)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    quantiles = ranks / max(len(ranks) - 1, 1)
    # Map quantiles to reference values
    ref_indices = (quantiles * (len(ref_sorted_vals) - 1)).astype(np.int64)
    ref_indices = np.clip(ref_indices, 0, len(ref_sorted_vals) - 1)
    matched_flat = ref_sorted_vals[ref_indices].astype(np.float32)
    return matched_flat.reshape(source.shape)
