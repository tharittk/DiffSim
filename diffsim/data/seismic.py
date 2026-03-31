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


def facies_to_ai(facies_map, ai_values=None):
    """
    Convert facies codes to acoustic impedance values.

    Args:
        facies_map: Integer array with facies codes {0: mud, 1: bank, 2: sand}
        ai_values: Dict mapping facies code → AI value. Uses DEFAULT_AI if None.

    Returns:
        AI map as float32 array (same shape as input)
    """
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
    reflectivity, f_dominant=25.0, dz=0.1, wavelet_duration=None
):
    """
    Compute synthetic seismic by convolving vertical reflectivity traces
    with a Ricker wavelet.

    Args:
        reflectivity: 3D reflectivity array (nx, ny, nz-1)
        f_dominant: Dominant wavelet frequency in Hz
        dz: Vertical sample spacing in meters (used as dt analog)
        wavelet_duration: Wavelet duration in meters. If None, auto-computed
                          from dominant frequency.

    Returns:
        Synthetic seismic volume (nx, ny, nz_out)
    """
    if wavelet_duration is None:
        wavelet_duration = 2.0 / f_dominant  # ~2 periods

    _, wavelet = ricker_wavelet(f_dominant, dz, wavelet_duration)

    nx, ny, nz = reflectivity.shape
    synthetic = np.zeros_like(reflectivity, dtype=np.float32)
    for ix in range(nx):
        for iy in range(ny):
            trace = reflectivity[ix, iy, :]
            synthetic[ix, iy, :] = np.convolve(trace, wavelet, mode="same")
    return synthetic


def compute_rms_window(synthetic_seismic, z_center, window_half):
    """
    Compute RMS amplitude from a 3D synthetic seismic volume over a depth window.

    Args:
        synthetic_seismic: 3D seismic array (nx, ny, nz)
        z_center: Center depth index for the RMS window
        window_half: Half-window size in samples

    Returns:
        2D RMS amplitude map (nx, ny)
    """
    nz = synthetic_seismic.shape[2]
    z_start = max(0, z_center - window_half)
    z_end = min(nz, z_center + window_half + 1)

    window_data = synthetic_seismic[:, :, z_start:z_end]
    rms = np.sqrt(np.mean(window_data**2, axis=2))
    return rms


def generate_rms_from_facies_3d(
    facies_block,
    z_target,
    ai_values=None,
    f_dominant=25.0,
    rms_window_half=5,
    noise_level=0.05,
    smooth_sigma=1.0,
    rng=None,
):
    """
    Full pipeline: 3D facies block → 2D RMS amplitude map at a target depth.

    Steps:
        1. Facies → Acoustic Impedance
        2. AI → Vertical reflectivity coefficients
        3. Reflectivity → Synthetic seismic (convolve with Ricker wavelet)
        4. Synthetic seismic → RMS amplitude in depth window
        5. Apply lateral smoothing and noise

    Args:
        facies_block: 3D array (nx, ny, nz) with facies codes {0, 1, 2}
        z_target: Depth index for RMS computation center
        ai_values: Dict of facies → AI values (uses defaults if None)
        f_dominant: Dominant frequency of Ricker wavelet in Hz
        rms_window_half: Half-window size for RMS computation (in samples)
        noise_level: Standard deviation of additive Gaussian noise
                     (relative to signal std)
        smooth_sigma: Lateral Gaussian smoothing sigma in grid cells
                      (simulates Fresnel zone)
        rng: numpy random Generator for reproducibility (None = global state)

    Returns:
        rms_map: 2D array (nx, ny) normalized to [-1, 1]
    """
    # 1. Facies → AI
    ai_block = facies_to_ai(facies_block, ai_values)

    # 2. AI → Reflectivity
    reflectivity = compute_reflectivity_vertical(ai_block)

    # 3. Reflectivity → Synthetic seismic
    dz = 0.1  # vertical sample spacing (matches hmax/30 ≈ 0.1m)
    synthetic = compute_synthetic_seismic_3d(reflectivity, f_dominant=f_dominant, dz=dz)

    # 4. Synthetic → RMS (clip z_target to reflectivity depth range)
    z_refl = min(z_target, reflectivity.shape[2] - 1)
    rms_map = compute_rms_window(synthetic, z_refl, rms_window_half)

    # 5. Lateral smoothing (simulates finite seismic resolution)
    if smooth_sigma > 0:
        rms_map = gaussian_filter(rms_map, sigma=smooth_sigma)

    # 6. Add band-limited noise
    if noise_level > 0:
        signal_std = np.std(rms_map)
        if signal_std > 1e-10:
            if rng is not None:
                noise = noise_level * signal_std * rng.standard_normal(rms_map.shape)
            else:
                noise = noise_level * signal_std * np.random.randn(*rms_map.shape)
            rms_map = rms_map + noise

    # Normalize to [-1, 1]
    rms_map = _normalize_to_range(rms_map)

    return rms_map.astype(np.float32)


def generate_rms_from_facies_2d(
    facies_map,
    ai_values=None,
    smooth_sigma=3.0,
    noise_level=0.05,
    rng=None,
):
    """
    Simplified pipeline: 2D facies map → pseudo-RMS amplitude map.

    For plan-view maps where vertical (3D) information is unavailable.
    Uses lateral convolution to approximate seismic resolution effects:
        1. Facies → Acoustic Impedance
        2. Apply Gaussian smoothing (simulates Fresnel zone blurring)
        3. Add band-limited noise
        4. Normalize to [-1, 1]

    The result is a smoothed, noisy version of the impedance map that
    preserves large-scale channel patterns while degrading fine details.

    Args:
        facies_map: 2D array (H, W) with facies codes {0, 1, 2}
        ai_values: Dict of facies → AI values (uses defaults if None)
        smooth_sigma: Gaussian smoothing sigma in grid cells
        noise_level: Relative noise level (fraction of signal std)
        rng: numpy random Generator for reproducibility (None = global state)

    Returns:
        rms_map: 2D array (H, W) normalized to [-1, 1]
    """
    # 1. Facies → AI
    ai_map = facies_to_ai(facies_map, ai_values)

    # 2. Smooth to simulate seismic resolution
    smoothed = gaussian_filter(ai_map.astype(np.float64), sigma=smooth_sigma)

    # 3. Add band-limited noise (smooth the noise too for realism)
    if noise_level > 0:
        if rng is not None:
            noise = rng.standard_normal(smoothed.shape)
        else:
            noise = np.random.randn(*smoothed.shape)
        noise = gaussian_filter(noise, sigma=max(smooth_sigma / 2, 0.5))
        signal_std = np.std(smoothed)
        if signal_std > 1e-10:
            smoothed = smoothed + noise_level * signal_std * noise

    # 4. Normalize to [-1, 1]
    rms_map = _normalize_to_range(smoothed)

    return rms_map.astype(np.float32)


def _normalize_to_range(arr, vmin=-1.0, vmax=1.0):
    """Normalize array to [vmin, vmax] range."""
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min < 1e-10:
        return np.full_like(arr, (vmin + vmax) / 2, dtype=np.float64)
    return (arr - arr_min) / (arr_max - arr_min) * (vmax - vmin) + vmin
