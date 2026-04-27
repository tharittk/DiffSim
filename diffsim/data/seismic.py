"""
Seismic forward modeling for generating synthetic RMS amplitude attributes.

Pipeline:
    1. Facies map → Acoustic Impedance (AI) map
    2. AI → Reflectivity coefficients (vertical impedance contrasts)
    3. Reflectivity → Synthetic seismic (convolution with Ricker wavelet)
    4. Synthetic seismic → RMS amplitude attribute

Supports both:
    - 3D block processing: physically rigorous vertical reflectivity + convolution
    - 2D plan-view processing: simplified lateral convolution approach (fallback)
"""

import numpy as np
from scipy.ndimage import gaussian_filter

# Default acoustic impedance values (g/cc * m/s)
# Typical values for shallow clastic sediments
DEFAULT_AI = {
    0: 10000.0,  # Mud (overbank): high impedance
    1: 8500.0,  # Bank (levee): moderate impedance
    2: 7000.0,  # Sand (channel fill): low impedance
}

# Per-facies rock property distributions: {facies_code: {"rhob": (mean, std), "vp": (mean, std)}}
# rhob in g/cc, vp in m/s.  AI = rhob * vp.
DEFAULT_ROCK_PROPERTIES = {
    0: {"rhob": (2.40, 0.05), "vp": (4200.0, 150.0)},  # Mud (overbank)
    1: {"rhob": (2.25, 0.05), "vp": (3700.0, 150.0)},  # Bank (levee)
    2: {"rhob": (2.10, 0.05), "vp": (3200.0, 150.0)},  # Sand (channel fill)
}


def ricker_wavelet(f_dominant, dt, duration):
    """
    Generate a Ricker wavelet (Mexican hat / second derivative of Gaussian).

    Args:
        f_dominant: Dominant frequency in Hz
        dt: Time/depth sampling interval in seconds/meters
        duration: Total wavelet duration in seconds/meters

    Returns:
        t: Time/depth vector
        wavelet: Ricker wavelet amplitudes (normalized to unit peak)
    """
    t = np.arange(-duration / 2, duration / 2 + dt, dt)
    pi2 = (np.pi * f_dominant) ** 2
    u = pi2 * t**2
    wavelet = (1.0 - 2.0 * u) * np.exp(-u)
    return t, wavelet


def facies_to_ai(facies_map, ai_values=None, rock_properties=None, rng=None):
    """
    Convert facies codes to acoustic impedance values.

    Two modes:
        - Deterministic (default): ``ai_values`` maps facies code → constant AI.
        - Stochastic: ``rock_properties`` maps facies code →
          ``{"rhob": (mean, std), "vp": (mean, std)}``.
          Per-voxel AI = rhob * vp sampled from normal distributions.

    When ``rock_properties`` is provided it takes precedence over ``ai_values``.

    Args:
        facies_map: Integer array with facies codes {0: mud, 1: bank, 2: sand}
        ai_values: Dict mapping facies code → AI value. Uses DEFAULT_AI if None.
        rock_properties: Dict mapping facies code → {"rhob": (mean, std),
                         "vp": (mean, std)}. If provided, AI is sampled
                         stochastically per voxel.
        rng: numpy ``Generator`` for reproducibility (used only in stochastic
             mode; ``None`` creates a fresh default generator).

    Returns:
        AI map as float32 array (same shape as input)
    """
    if rock_properties is not None:
        if rng is None:
            rng = np.random.default_rng()
        ai_map = np.zeros_like(facies_map, dtype=np.float32)
        for code, props in rock_properties.items():
            mask = facies_map == code
            n_voxels = int(mask.sum())
            if n_voxels == 0:
                continue
            rhob_mean, rhob_std = props["rhob"]
            vp_mean, vp_std = props["vp"]
            rhob = rng.normal(rhob_mean, rhob_std, size=n_voxels)
            vp = rng.normal(vp_mean, vp_std, size=n_voxels)
            ai_map[mask] = (rhob * vp).astype(np.float32)
        return ai_map

    # Deterministic fallback
    if ai_values is None:
        ai_values = DEFAULT_AI
    ai_map = np.zeros_like(facies_map, dtype=np.float32)
    for code, ai in ai_values.items():
        ai_map[facies_map == code] = ai
    return ai_map


def compute_reflectivity_vertical(ai_block):
    """
    Compute reflectivity coefficients from vertical AI contrasts in a 3D block.

    R[x,y,z] = (AI[x,y,z+1] - AI[x,y,z]) / (AI[x,y,z+1] + AI[x,y,z])

    Args:
        ai_block: 3D AI array of shape (nx, ny, nz)

    Returns:
        Reflectivity array of shape (nx, ny, nz-1)
    """
    ai_above = ai_block[:, :, :-1]
    ai_below = ai_block[:, :, 1:]
    denominator = ai_above + ai_below
    denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    reflectivity = (ai_below - ai_above) / denominator
    return reflectivity


def compute_synthetic_seismic_3d(
    reflectivity,
    f_dominant=25.0,
    dz=0.2,
    wavelet_duration=None,
    velocity=2500.0,
):
    """
    Compute synthetic seismic by convolving vertical reflectivity traces
    with a Ricker wavelet.

    Args:
        reflectivity: 3D reflectivity array (nx, ny, nz-1)
        f_dominant: Dominant wavelet frequency in Hz (time domain)
        dz: Vertical sample spacing in meters (used as dt analog)
        wavelet_duration: Wavelet duration in meters (depth domain). If None,
                  auto-computed from the depth-domain dominant
                  frequency.
        velocity: Average propagation velocity in m/s used to convert
              time-domain frequency (Hz) to depth-domain frequency
              (cycles/m), using t = 2*z / v.

    Returns:
        Synthetic seismic volume (nx, ny, nz_out)
    """
    if f_dominant <= 0:
        raise ValueError("f_dominant must be positive")
    if dz <= 0:
        raise ValueError("dz must be positive")
    if velocity <= 0:
        raise ValueError("velocity must be positive")

    # Convert user-facing time-domain Hz to depth-domain cycles/m.
    f_depth = (2.0 * f_dominant) / velocity

    if wavelet_duration is None:
        wavelet_duration = 2.0 / f_depth  # ~2 periods in depth domain
    elif wavelet_duration <= 0:
        raise ValueError("wavelet_duration must be positive")

    _, wavelet = ricker_wavelet(f_depth, dz, wavelet_duration)

    nx, ny, nz = reflectivity.shape
    synthetic = np.zeros_like(reflectivity, dtype=np.float32)
    for ix in range(nx):
        for iy in range(ny):
            trace = reflectivity[ix, iy, :]
            conv = np.convolve(trace, wavelet, mode="same")
            if conv.shape[0] != nz:
                # numpy's "same" returns max(len(trace), len(wavelet)); center-crop to trace length.
                start = (conv.shape[0] - nz) // 2
                conv = conv[start : start + nz]
            synthetic[ix, iy, :] = conv
    return synthetic


def compute_rms_cube(synthetic_seismic, window_half):
    """
    Compute a depth-wise RMS cube from a 3D synthetic seismic volume.

    For each depth index z, RMS is computed over the vertical window
    [z-window_half, z+window_half]. Near boundaries, the window is clipped.

    Args:
        synthetic_seismic: 3D seismic array (nx, ny, nz)
        window_half: Half-window size in samples

    Returns:
        3D RMS cube (nx, ny, nz), unnormalized
    """
    if window_half < 0:
        raise ValueError("window_half must be >= 0")

    nx, ny, nz = synthetic_seismic.shape
    sq = synthetic_seismic.astype(np.float64) ** 2
    csum = np.cumsum(sq, axis=2)

    rms_cube = np.empty((nx, ny, nz), dtype=np.float32)
    for z in range(nz):
        z_start = max(0, z - window_half)
        z_end = min(nz, z + window_half + 1)
        window_sum = csum[:, :, z_end - 1]
        if z_start > 0:
            window_sum = window_sum - csum[:, :, z_start - 1]
        window_len = z_end - z_start
        rms_cube[:, :, z] = np.sqrt(window_sum / float(window_len))

    return rms_cube


def generate_rms_from_facies_3d(
    facies_block,
    ai_values=None,
    rock_properties=None,
    f_dominant=25.0,
    rms_window_half=5,
    noise_level=0.05,
    smooth_sigma=1.0,
    rng=None,
    velocity=2500.0,
):
    """
    Full pipeline: 3D facies block → 3D RMS amplitude cube.

    Steps:
        1. Facies → Acoustic Impedance
        2. AI → Vertical reflectivity coefficients
        3. Reflectivity → Synthetic seismic (convolve with Ricker wavelet)
        4. Synthetic seismic → RMS cube (depth-windowed)
        5. Apply lateral smoothing and noise

    Args:
        facies_block: 3D array (nx, ny, nz) with facies codes {0, 1, 2}
        ai_values: Dict of facies → AI values (used only in deterministic
                   mode, i.e. when ``rock_properties`` is None).
        rock_properties: Dict of facies → {"rhob": (mean, std), "vp": (mean, std)}.
                         When provided, AI is sampled stochastically per voxel
                         as ``rhob * vp``.  Defaults to ``DEFAULT_ROCK_PROPERTIES``
                         when both this and ``ai_values`` are None.
        f_dominant: Dominant frequency of Ricker wavelet in Hz
        rms_window_half: Half-window size for RMS computation (in samples)
        noise_level: Standard deviation of additive Gaussian noise
                     (relative to signal std)
        smooth_sigma: Lateral Gaussian smoothing sigma in grid cells
                      (simulates Fresnel zone)
        rng: numpy random Generator for reproducibility (None = global state)
        velocity: Average propagation velocity in m/s used for the internal
              time-to-depth conversion of f_dominant.

    Returns:
        rms_cube: 3D array (nx, ny, nz-1), unnormalized
    """
    # check if the facies block contains valid codes
    valid_codes = set(DEFAULT_AI.keys())
    unique_codes = set(np.unique(facies_block).tolist())
    if not unique_codes.issubset(valid_codes):
        raise ValueError(
            f"Facies block has unexpected codes: {unique_codes - valid_codes}. "
            f"Expected subset of {valid_codes}."
        )
    # Default to stochastic rock-property sampling when neither is specified.
    if ai_values is None and rock_properties is None:
        rock_properties = DEFAULT_ROCK_PROPERTIES

    # 1. Facies → AI
    ai_block = facies_to_ai(
        facies_block, ai_values=ai_values, rock_properties=rock_properties, rng=rng
    )

    # 2. AI → Reflectivity
    reflectivity = compute_reflectivity_vertical(ai_block)

    # 3. Reflectivity → Synthetic seismic
    dz = 0.1  # vertical sample spacing (matches hmax/30 ≈ 0.1m)
    synthetic = compute_synthetic_seismic_3d(
        reflectivity,
        f_dominant=f_dominant,
        dz=dz,
        velocity=velocity,
    )

    # 4. Synthetic → RMS cube
    rms_cube = compute_rms_cube(synthetic, rms_window_half)

    # 5. Lateral smoothing (simulates finite seismic resolution)
    if smooth_sigma > 0:
        rms_cube = gaussian_filter(rms_cube, sigma=(smooth_sigma, smooth_sigma, 0.0))

    # 6. Add band-limited noise
    if noise_level > 0:
        signal_std = np.std(rms_cube)
        if signal_std > 1e-10:
            if rng is not None:
                noise = rng.standard_normal(rms_cube.shape)
            else:
                noise = np.random.randn(*rms_cube.shape)
            if smooth_sigma > 0:
                noise = gaussian_filter(
                    noise,
                    sigma=(
                        max(smooth_sigma / 2.0, 0.5),
                        max(smooth_sigma / 2.0, 0.5),
                        0.0,
                    ),
                )
            rms_cube = rms_cube + noise_level * signal_std * noise

    return rms_cube.astype(np.float32)


def normalize_rms(arr, vmin=-1.0, vmax=1.0):
    """Normalize array to [vmin, vmax] range."""
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min < 1e-10:
        return np.full_like(arr, (vmin + vmax) / 2, dtype=np.float32)
    return (arr - arr_min) / (arr_max - arr_min) * (vmax - vmin) + vmin
