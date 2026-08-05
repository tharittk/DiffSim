"""
Inference utilities for conditional diffusion models.

Supports:
    - Arbitrary input sizes (e.g., 256x256) for models trained on 64x64
    - Multi-sample inference with per-facies probability computation
"""

import numpy as np
import torch
from tqdm import tqdm

from diffsim.data.flumy_generator import denormalize_facies, FACIES_NORMALIZED
from diffsim.data.seismic import normalize_rms


def prepare_conditioning(rms_map, device="cuda"):
    """
    Prepare a full-resolution RMS map as conditioning tensor.

    Since the network uses only convolutional layers, it can accept
    any spatial resolution at inference time (not restricted to training size).

    Args:
        rms_map: 2D numpy array (H, W) - raw RMS amplitude values
        device: torch device

    Returns:
        cond_image: Tensor (1, 1, H, W) normalized to [-1, 1]
    """
    rms_norm = normalize_rms(rms_map, vmin=-1.0, vmax=1.0)
    cond_tensor = torch.from_numpy(rms_norm[np.newaxis, np.newaxis, :, :]).float()
    return cond_tensor.to(device)


@torch.no_grad()
def run_inference(network, cond_image, ddim_steps=50, eta=0.0):
    """
    Run a single diffusion inference at arbitrary spatial resolution.

    The model is fully convolutional, so it generalizes to any input size.

    Args:
        network: Trained Network instance (with noise schedule set to 'test')
        cond_image: Conditioning tensor (1, C_cond, H, W)
        ddim_steps: Number of DDIM sampling steps
        eta: DDIM stochasticity (0 = deterministic)

    Returns:
        output: Generated tensor (1, 1, H, W) in normalized [-1, 1] range
    """
    y_t, _ = network.restoration_ddim(
        y_cond=cond_image,
        ddim_steps=ddim_steps,
        eta=eta,
        sample_num=1,
    )
    return y_t


@torch.no_grad()
def multi_inference_probabilities(
    network,
    cond_image,
    n_samples=10,
    ddim_steps=50,
    eta=0.0,
    facies_codes=(0, 1, 2, 3, 4, 5),
):
    """
    Run multiple diffusion inferences and compute per-facies probability maps.

    Each inference produces a different realization due to different initial
    noise. The ensemble is used to estimate P(facies=k | RMS) at each pixel.

    Args:
        network: Trained Network instance (with noise schedule set to 'test')
        cond_image: Conditioning tensor (1, C_cond, H, W) - can be any size
        n_samples: Number of independent inferences to run
        ddim_steps: Number of DDIM steps per inference
        eta: DDIM stochasticity (0 = deterministic, >0 adds diversity)
        facies_codes: Tuple of integer facies codes to compute probabilities for

    Returns:
        probabilities: Dict {facies_code: np.ndarray (H, W)} with values in [0, 1]
                       representing P(facies=code | conditioning) at each pixel.
        samples: np.ndarray (n_samples, H, W) of integer facies predictions
    """
    _, _, H, W = cond_image.shape
    samples = np.empty((n_samples, H, W), dtype=np.int8)

    for i in tqdm(range(n_samples), desc="Multi-inference"):
        output = run_inference(network, cond_image, ddim_steps=ddim_steps, eta=eta)
        # Convert normalized output to integer facies codes
        output_np = output.squeeze().cpu().numpy()
        facies_int = denormalize_facies(output_np)
        samples[i] = facies_int

    # Compute per-facies probability (frequency of each code across samples)
    probabilities = {}
    for code in facies_codes:
        prob_map = (samples == code).astype(np.float32).mean(axis=0)
        probabilities[code] = prob_map

    return probabilities, samples


@torch.no_grad()
def inference_full_resolution(
    network,
    rms_map,
    n_samples=10,
    ddim_steps=50,
    eta=0.0,
    device="cuda",
):
    """
    End-to-end inference on a full-resolution RMS map (e.g., 256x256).

    Runs multiple diffusion samples and returns facies probability maps.
    The model trained on 64x64 patches works on 256x256 because it uses
    only convolutional kernels that are translation-equivariant.

    Args:
        network: Trained Network instance
        rms_map: 2D numpy array (H, W) - raw RMS amplitude map at original resolution
        n_samples: Number of independent realizations
        ddim_steps: DDIM sampling steps
        eta: Stochasticity parameter (use >0 for diversity in ensemble)
        device: torch device string

    Returns:
        probabilities: Dict {facies_code: np.ndarray (H, W)} probability maps
        samples: np.ndarray (n_samples, H, W) integer facies predictions
        most_likely: np.ndarray (H, W) most probable facies at each pixel
    """
    # Set test-time noise schedule
    network.set_new_noise_schedule(device=torch.device(device), phase="test")
    network.eval()

    # Prepare conditioning at full resolution
    cond_image = prepare_conditioning(rms_map, device=device)

    # Run multi-sample inference
    probabilities, samples = multi_inference_probabilities(
        network,
        cond_image,
        n_samples=n_samples,
        ddim_steps=ddim_steps,
        eta=eta,
    )

    # Compute most likely facies: map argmax index back to actual facies code
    sorted_codes = sorted(probabilities.keys())
    prob_stack = np.stack(
        [probabilities[code] for code in sorted_codes], axis=-1
    )
    most_likely = np.array(sorted_codes, dtype=np.int8)[
        np.argmax(prob_stack, axis=-1)
    ]

    return probabilities, samples, most_likely
