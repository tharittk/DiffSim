"""
Diffusion utilities - beta schedules, sampling functions for unconditional diffusion.

Supports both 2D and 3D data.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Linear beta schedule from beta_start to beta_end.
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Quadratic beta schedule.
    """
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Sigmoid beta schedule.
    """
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def get_beta_schedule(schedule_type, timesteps, **kwargs):
    """
    Get beta schedule by name.

    Args:
        schedule_type: One of 'linear', 'cosine', 'quadratic', 'sigmoid'
        timesteps: Number of diffusion timesteps
        **kwargs: Additional arguments for the schedule function

    Returns:
        Beta schedule tensor
    """
    schedules = {
        'linear': linear_beta_schedule,
        'cosine': cosine_beta_schedule,
        'quadratic': quadratic_beta_schedule,
        'sigmoid': sigmoid_beta_schedule,
    }
    if schedule_type not in schedules:
        raise ValueError(f"Unknown schedule type: {schedule_type}. Choose from {list(schedules.keys())}")
    return schedules[schedule_type](timesteps, **kwargs)


class Diffusion:
    """
    Diffusion process handler for unconditional diffusion models.

    Supports both 2D and 3D data by computing noise schedules
    and providing sampling functions.

    Args:
        timesteps: Number of diffusion timesteps
        beta_schedule: Type of beta schedule ('linear', 'cosine', etc.)
        **schedule_kwargs: Additional arguments for the beta schedule
    """

    def __init__(self, timesteps=1000, beta_schedule='linear', **schedule_kwargs):
        self.timesteps = timesteps

        # Define beta schedule
        self.betas = get_beta_schedule(beta_schedule, timesteps, **schedule_kwargs)

        # Define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def _extract(self, a, t, x_shape):
        """Extract values from a at indices t, reshape for broadcasting."""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: add noise to x_start at timestep t.

        Args:
            x_start: Original data [B, C, ...]
            t: Timesteps [B]
            noise: Optional pre-generated noise

        Returns:
            Noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1"):
        """
        Compute loss for training the denoising model.

        Args:
            denoise_model: The model that predicts noise
            x_start: Original data [B, C, ...]
            t: Timesteps [B]
            noise: Optional pre-generated noise
            loss_type: One of 'l1', 'l2', 'huber'

        Returns:
            Loss value
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError(f"Loss type {loss_type} not implemented")

        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """
        Single step of reverse diffusion: sample x_{t-1} from x_t.

        Args:
            model: Denoising model
            x: Current noisy data
            t: Current timestep tensor
            t_index: Current timestep index

        Returns:
            Denoised sample at t-1
        """
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, return_all_steps=False):
        """
        Full reverse diffusion loop: generate samples from noise.

        Args:
            model: Denoising model
            shape: Output shape (batch_size, channels, ...)
            return_all_steps: If True, return all intermediate steps

        Returns:
            Generated samples, optionally with all intermediate steps
        """
        device = next(model.parameters()).device

        b = shape[0]
        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling', total=self.timesteps):
            img = self.p_sample(
                model, img,
                torch.full((b,), i, device=device, dtype=torch.long),
                i
            )
            if return_all_steps:
                imgs.append(img.cpu().numpy())

        if return_all_steps:
            return imgs
        return img

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3, return_all_steps=False):
        """
        Generate samples using the model.

        Args:
            model: Denoising model
            image_size: Size of output (int for square 2D, tuple for 3D)
            batch_size: Number of samples to generate
            channels: Number of channels
            return_all_steps: If True, return all intermediate steps

        Returns:
            Generated samples
        """
        if isinstance(image_size, int):
            shape = (batch_size, channels, image_size, image_size)
        else:
            shape = (batch_size, channels, *image_size)

        return self.p_sample_loop(model, shape, return_all_steps)


    @torch.no_grad()
    def ddim_sample(self, model, x, t, t_prev, eta=0.0):
        """
        Single DDIM sampling step.

        Args:
            model: Denoising model
            x: Current noisy data
            t: Current timestep
            t_prev: Previous timestep
            eta: DDIM stochasticity parameter (0 = deterministic)

        Returns:
            Sample at t_prev
        """
        alpha_t = self._extract(self.alphas_cumprod, t, x.shape)
        alpha_t_prev = self._extract(self.alphas_cumprod, t_prev, x.shape) if t_prev[0] >= 0 else torch.ones_like(alpha_t)

        # Predict noise
        predicted_noise = model(x, t)

        # Predict x_0
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)

        # Compute sigma
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * predicted_noise

        # Random noise
        noise = torch.randn_like(x) if eta > 0 else 0

        # x_{t-1}
        x_prev = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma_t * noise
        return x_prev

    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, ddim_steps=50, eta=0.0, return_all_steps=False):
        """
        DDIM sampling loop for faster generation.

        Args:
            model: Denoising model
            shape: Output shape (batch_size, channels, ...)
            ddim_steps: Number of DDIM steps (fewer than full timesteps)
            eta: DDIM stochasticity (0 = deterministic, 1 = DDPM-like)
            return_all_steps: If True, return all intermediate steps

        Returns:
            Generated samples
        """
        device = next(model.parameters()).device
        b = shape[0]

        # Create DDIM timestep sequence
        c = self.timesteps // ddim_steps
        ddim_timesteps = list(range(0, self.timesteps, c))

        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(len(ddim_timesteps))), desc='DDIM Sampling', total=len(ddim_timesteps)):
            t = torch.full((b,), ddim_timesteps[i], device=device, dtype=torch.long)
            t_prev = torch.full((b,), ddim_timesteps[i-1] if i > 0 else -1, device=device, dtype=torch.long)
            img = self.ddim_sample(model, img, t, t_prev, eta=eta)

            if return_all_steps:
                imgs.append(img.cpu().numpy())

        if return_all_steps:
            return imgs
        return img

    @torch.no_grad()
    def sample_ddim(self, model, image_size, batch_size=16, channels=3, ddim_steps=50, eta=0.0, return_all_steps=False):
        """
        Generate samples using DDIM (faster than DDPM).

        Args:
            model: Denoising model
            image_size: Size of output (int for square 2D, tuple for 3D)
            batch_size: Number of samples to generate
            channels: Number of channels
            ddim_steps: Number of DDIM steps
            eta: DDIM stochasticity (0 = deterministic)
            return_all_steps: If True, return all intermediate steps

        Returns:
            Generated samples
        """
        if isinstance(image_size, int):
            shape = (batch_size, channels, image_size, image_size)
        else:
            shape = (batch_size, channels, *image_size)

        return self.ddim_sample_loop(model, shape, ddim_steps, eta, return_all_steps)


# Convenience function for backwards compatibility
def sample(model, image_size, batch_size=16, channels=3, timesteps=1000,
           beta_schedule='linear', return_all_steps=True):
    """
    Generate samples using the model with default diffusion settings.

    Args:
        model: Denoising model
        image_size: Size of output (int for square 2D, tuple for 3D)
        batch_size: Number of samples to generate
        channels: Number of channels
        timesteps: Number of diffusion timesteps
        beta_schedule: Type of beta schedule
        return_all_steps: If True, return all intermediate steps

    Returns:
        Generated samples
    """
    diffusion = Diffusion(timesteps=timesteps, beta_schedule=beta_schedule)
    return diffusion.sample(model, image_size, batch_size, channels, return_all_steps)
