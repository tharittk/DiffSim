"""
Network wrapper for conditional diffusion models.

Handles noise schedules, sampling, and training for guided diffusion.
"""

import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from .base_network import BaseNetwork


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape=(1, 1, 1, 1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    """
    Create beta schedule for diffusion process.

    Args:
        schedule: Schedule type ('quad', 'linear', 'warmup10', 'warmup50', 'const', 'jsd', 'cosine')
        n_timestep: Number of timesteps
        linear_start: Starting beta value for linear schedules
        linear_end: Ending beta value for linear schedules
        cosine_s: Offset for cosine schedule

    Returns:
        Beta schedule as numpy array or torch tensor
    """
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


class Network(BaseNetwork):
    """
    Network wrapper for conditional diffusion models.

    Handles the diffusion process including noise scheduling,
    forward/reverse diffusion, and restoration (inpainting).

    Args:
        unet: Dictionary of UNet configuration parameters
        beta_schedule: Dictionary with 'train' and optionally 'test' schedules
        module_name: Module to use for UNet ('sr3' or 'guided_diffusion')
        predict_type: Prediction type - 'epsilon' (predict noise) or 'x_start' (predict x0)
        **kwargs: Additional arguments for BaseNetwork
    """

    def __init__(self, unet, beta_schedule, module_name='guided_diffusion', predict_type='epsilon', **kwargs):
        super(Network, self).__init__(**kwargs)

        if module_name == 'sr3':
            from ..models.guided_diffusion.unet import UNet
        elif module_name == 'guided_diffusion':
            from ..models.guided_diffusion.unet import UNet
        elif module_name == 'guided_diffusion_3d':
            from ..models.guided_diffusion.unet3d import UNet3D as UNet
        else:
            raise ValueError(f"Unknown module_name: {module_name}")

        self.denoise_fn = UNet(**unet)
        self.beta_schedule = beta_schedule
        self.predict_type = predict_type  # 'epsilon' or 'x_start'
        self.loss_fn = None

    def set_loss(self, loss_fn):
        """Set the loss function for training."""
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        """
        Initialize noise schedule parameters.

        Args:
            device: Device to put tensors on
            phase: 'train' or 'test' to select schedule
        """
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # Log calculation clipped because the posterior variance is 0 at the beginning
        self.register_buffer('posterior_log_variance_clipped',
                             to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',
                             to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        """Predict x_0 from x_t and predicted noise."""
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        """Compute posterior mean and variance."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        """Compute model mean and variance for reverse process."""
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        model_output = self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level)

        if self.predict_type == 'epsilon':
            # Model predicts noise, compute x_0 from noise
            y_0_hat = self.predict_start_from_noise(y_t, t=t, noise=model_output)
        elif self.predict_type == 'x_start':
            # Model directly predicts x_0
            y_0_hat = model_output
        else:
            raise ValueError(f"Unknown predict_type: {self.predict_type}")

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        """Forward diffusion: add noise to y_0."""
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        """Single reverse diffusion step."""
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8):
        """
        Perform conditional generation / inpainting.

        Args:
            y_cond: Conditioning input
            y_t: Optional starting noise (defaults to random)
            y_0: Ground truth for masked regions (for inpainting)
            mask: Binary mask (1 = generate, 0 = keep from y_0)
            sample_num: Number of intermediate samples to return

        Returns:
            y_t: Final generated sample
            ret_arr: Array of intermediate samples
        """
        b, *_ = y_cond.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must be greater than sample_num'
        sample_inter = (self.num_timesteps // sample_num)

        y_t = default(y_t, lambda: torch.randn_like(y_cond[:, :1, ...]))
        ret_arr = y_t
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling', total=self.num_timesteps):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=y_cond)
            if mask is not None:
                y_t = y_0 * (1. - mask) + mask * y_t
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
        return y_t, ret_arr

    @torch.no_grad()
    def ddim_p_sample(self, y_t, t, t_prev, clip_denoised=True, y_cond=None, eta=0.0):
        """
        Single DDIM reverse diffusion step.

        Args:
            y_t: Current noisy sample
            t: Current timestep
            t_prev: Previous timestep
            clip_denoised: Whether to clip predicted x_0
            y_cond: Conditioning input
            eta: DDIM stochasticity (0 = deterministic, 1 = DDPM-like)

        Returns:
            Sample at t_prev
        """
        b = y_t.shape[0]
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        model_output = self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level)

        # Get alpha values (gammas = alpha_cumprod)
        alpha_t = extract(self.gammas, t, y_t.shape)
        alpha_t_prev = extract(self.gammas, t_prev, y_t.shape) if t_prev[0] >= 0 else torch.ones_like(alpha_t)

        if self.predict_type == 'epsilon':
            # Model predicts noise, compute x_0 from noise
            pred_x0 = self.predict_start_from_noise(y_t, t=t, noise=model_output)
            predicted_noise = model_output
        elif self.predict_type == 'x_start':
            # Model directly predicts x_0
            pred_x0 = model_output
            # Compute noise from x_0 prediction
            predicted_noise = (y_t - torch.sqrt(alpha_t) * pred_x0) / torch.sqrt(1 - alpha_t)
        else:
            raise ValueError(f"Unknown predict_type: {self.predict_type}")

        if clip_denoised:
            pred_x0.clamp_(-1., 1.)

        # Compute sigma for DDIM
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * predicted_noise

        # Random noise
        noise = torch.randn_like(y_t) if eta > 0 else 0

        # x_{t-1}
        y_prev = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma_t * noise
        return y_prev

    @torch.no_grad()
    def restoration_ddim(self, y_cond, y_t=None, y_0=None, mask=None, ddim_steps=50, eta=0.0, sample_num=8):
        """
        Perform conditional generation / inpainting using DDIM (faster).

        Args:
            y_cond: Conditioning input
            y_t: Optional starting noise (defaults to random)
            y_0: Ground truth for masked regions (for inpainting)
            mask: Binary mask (1 = generate, 0 = keep from y_0)
            ddim_steps: Number of DDIM steps (fewer than full timesteps)
            eta: DDIM stochasticity (0 = deterministic, 1 = DDPM-like)
            sample_num: Number of intermediate samples to return

        Returns:
            y_t: Final generated sample
            ret_arr: Array of intermediate samples
        """
        b, *_ = y_cond.shape

        # Create DDIM timestep sequence
        c = self.num_timesteps // ddim_steps
        ddim_timesteps = list(range(0, self.num_timesteps, c))

        assert ddim_steps > sample_num, 'ddim_steps must be greater than sample_num'
        sample_inter = ddim_steps // sample_num

        y_t = default(y_t, lambda: torch.randn_like(y_cond[:, :1, ...]))
        ret_arr = y_t

        for i, step_idx in enumerate(tqdm(reversed(range(len(ddim_timesteps))), desc='DDIM Sampling', total=len(ddim_timesteps))):
            t = torch.full((b,), ddim_timesteps[step_idx], device=y_cond.device, dtype=torch.long)
            t_prev = torch.full((b,), ddim_timesteps[step_idx - 1] if step_idx > 0 else -1, device=y_cond.device, dtype=torch.long)

            y_t = self.ddim_p_sample(y_t, t, t_prev, y_cond=y_cond, eta=eta)

            if mask is not None:
                y_t = y_0 * (1. - mask) + mask * y_t

            if (len(ddim_timesteps) - 1 - step_idx) % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)

        return y_t, ret_arr

    def forward(self, y_0, y_cond=None, mask=None, noise=None):
        """
        Forward pass for training.

        Args:
            y_0: Ground truth images
            y_cond: Conditioning input
            mask: Optional mask for masked training
            noise: Optional pre-generated noise

        Returns:
            Training loss
        """
        # Sampling from p(gammas)
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = extract(self.gammas, t - 1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2 - gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))

        # Determine shape for broadcasting
        if y_0.dim() == 4:
            sample_gammas_shaped = sample_gammas.view(-1, 1, 1, 1)
        else:
            sample_gammas_shaped = sample_gammas.view(-1, 1, 1, 1, 1)

        y_noisy = self.q_sample(y_0=y_0, sample_gammas=sample_gammas_shaped, noise=noise)

        if mask is not None:
            model_output = self.denoise_fn(
                torch.cat([y_cond, y_noisy * mask + (1. - mask) * y_0], dim=1),
                sample_gammas
            )
            if self.predict_type == 'epsilon':
                # Model predicts noise
                loss = self.loss_fn(mask * noise, mask * model_output)
            elif self.predict_type == 'x_start':
                # Model predicts x_0 directly
                loss = self.loss_fn(mask * y_0, mask * model_output)
            else:
                raise ValueError(f"Unknown predict_type: {self.predict_type}")
        else:
            model_output = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
            if self.predict_type == 'epsilon':
                # Model predicts noise
                loss = self.loss_fn(noise, model_output)
            elif self.predict_type == 'x_start':
                # Model predicts x_0 directly
                loss = self.loss_fn(y_0, model_output)
            else:
                raise ValueError(f"Unknown predict_type: {self.predict_type}")
        return loss
