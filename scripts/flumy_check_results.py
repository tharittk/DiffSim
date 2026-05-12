"""
QC utilities for comparing conditional diffusion model results.

Supports two scenarios:
    - Scenario A (epsilon): model predicts noise, subtract to get output
    - Scenario B (x_start): model predicts output image directly

Both models share the same architecture (UNet) and differ only in predict_type.

Usage (notebook-friendly):
    from flumy_check_results import ResultsQC

    qc = ResultsQC(
        run_epsilon="20260507_141532",
        run_xstart="20260510_112719",
        base_dir="/mnt/sda_data/tharitt/diffsim",
    )

    # Plot training loss curves
    qc.plot_losses()

    # Run inference on a test sample
    qc.infer_and_compare(sample_index=0)
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from diffsim.core.network import Network
from diffsim.data.flumy_dataset import FlumyDataset
from diffsim.data.flumy_generator import (
    denormalize_facies,
    FACIES_NAMES,
    FACIES_NORMALIZED,
)

# ---------------------------------------------------------------------------
# Facies colormap
# ---------------------------------------------------------------------------
FACIES_COLORS = {0: "#8B4513", 1: "#DAA520", 2: "#FFD700"}  # mud  # bank  # sand
FACIES_CMAP = mcolors.ListedColormap([FACIES_COLORS[i] for i in range(3)])
FACIES_NORM = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], FACIES_CMAP.N)


# ---------------------------------------------------------------------------
# TensorBoard log reader
# ---------------------------------------------------------------------------


def read_tensorboard_logs(log_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Read scalar data from TensorBoard event files.

    Args:
        log_dir: Path to directory containing tfevents files.

    Returns:
        Dict mapping tag name -> list of (step, value) tuples.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        raise ImportError(
            "tensorboard is required to read logs. "
            "Install with: pip install tensorboard"
        )

    ea = EventAccumulator(str(log_dir))
    ea.Reload()

    scalars = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        scalars[tag] = [(e.step, e.value) for e in events]
    return scalars


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


def load_model(
    model_dir: str,
    config_path: Optional[str] = None,
    checkpoint: str = "best_model.pth",
    device: Optional[torch.device] = None,
) -> Network:
    """
    Load a trained conditional diffusion Network from a model directory.

    Args:
        model_dir: Path to model directory (contains config.json + .pth files).
        config_path: Optional override for config file (defaults to model_dir/config.json).
        checkpoint: Filename of checkpoint to load. Use "best_model.pth" for the
                    best EMA model or "checkpoint_epoch_XXX.pth" for a specific epoch.
        device: Target device (defaults to CUDA if available).

    Returns:
        Network instance with weights loaded, in eval mode, on the target device.
    """
    model_dir = Path(model_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    cfg_path = Path(config_path) if config_path else model_dir / "config.json"
    with open(cfg_path) as f:
        config = json.load(f)

    cond = config["conditional"]

    unet_config = {
        "image_size": config["image_size"],
        "in_channel": cond["in_channel"],
        "out_channel": cond["out_channel"],
        "inner_channel": cond["inner_channel"],
        "channel_mults": cond["channel_mults"],
        "attn_res": cond["attn_res"],
        "res_blocks": cond["res_blocks"],
        "dropout": cond["dropout"],
    }

    module_name = cond.get("module_name", "guided_diffusion")
    predict_type = cond.get("predict_type", "epsilon")

    network = Network(
        unet=unet_config,
        beta_schedule=cond["beta_schedule"],
        module_name=module_name,
        predict_type=predict_type,
    )

    network.to(device)

    # Register noise schedule buffers BEFORE loading state dict,
    # so that EMA-saved buffers (gammas, etc.) match existing keys.
    network.set_new_noise_schedule(device=device, phase="test")

    # Load weights
    ckpt_path = model_dir / checkpoint
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Checkpoint may be a full dict (from periodic saves) or plain state_dict
    if isinstance(state, dict) and "model_state" in state:
        network.load_state_dict(state["model_state"])
    else:
        network.load_state_dict(state)

    network.eval()

    return network


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_inference(
    network: Network,
    cond_image: torch.Tensor,
    device: Optional[torch.device] = None,
    use_ddim: bool = True,
    ddim_steps: int = 50,
    eta: float = 0.0,
) -> np.ndarray:
    """
    Run conditional generation given an RMS conditioning image.

    Args:
        network: Loaded Network in eval mode.
        cond_image: Conditioning tensor of shape (1, 1, H, W), values in [-1, 1].
        device: Device (defaults to network device).
        use_ddim: Use DDIM sampling (faster) instead of full DDPM.
        ddim_steps: Number of DDIM steps.
        eta: DDIM stochasticity parameter.

    Returns:
        Generated facies image as int8 array (H, W) with codes {0, 1, 2}.
    """
    device = device or next(network.parameters()).device
    cond_image = cond_image.to(device)

    if use_ddim:
        output, _ = network.restoration_ddim(
            y_cond=cond_image, ddim_steps=ddim_steps, eta=eta, sample_num=4
        )
    else:
        output, _ = network.restoration(y_cond=cond_image, sample_num=4)

    # output shape: (1, 1, H, W) -> (H, W)
    output_np = output.squeeze().cpu().numpy()
    facies_int = denormalize_facies(output_np)
    return facies_int


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_facies(
    ax: plt.Axes,
    facies: np.ndarray,
    title: str = "",
    show_colorbar: bool = True,
):
    """Plot a facies map with the standard 3-class colormap."""
    im = ax.imshow(facies, cmap=FACIES_CMAP, norm=FACIES_NORM, origin="upper")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2], fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(["Mud", "Bank", "Sand"], fontsize=8)
    return im


def plot_rms(
    ax: plt.Axes, rms: np.ndarray, title: str = "RMS", show_colorbar: bool = True
):
    """Plot an RMS amplitude map."""
    im = ax.imshow(rms, cmap="seismic", origin="upper")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


# ---------------------------------------------------------------------------
# Main QC class
# ---------------------------------------------------------------------------


class ResultsQC:
    """
    Quality-control comparison of two conditional diffusion runs.

    Parameters
    ----------
    run_epsilon : str
        Timestamp of the epsilon-prediction run (e.g. "20260507_141532").
    run_xstart : str
        Timestamp of the x_start-prediction run (e.g. "20260510_112719").
    base_dir : str
        Root directory containing results/, model/, logs/ subdirectories.
    experiment : str
        Experiment name prefix (default "case1_flumy_conditional").
    device : torch.device, optional
        Compute device.
    """

    def __init__(
        self,
        run_epsilon: str,
        run_xstart: str,
        base_dir: str = "/mnt/sda_data/tharitt/diffsim",
        experiment: str = "case1_flumy_conditional",
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.base_dir = Path(base_dir)
        self.experiment = experiment

        self.run_ids = {"epsilon": run_epsilon, "x_start": run_xstart}
        self.labels = {"epsilon": "Predict Noise (ε)", "x_start": "Predict Image (x₀)"}

        # Paths
        self.model_dirs = {
            k: self.base_dir / "model" / experiment / v for k, v in self.run_ids.items()
        }
        self.log_dirs = {
            k: self.base_dir / "logs" / experiment / v for k, v in self.run_ids.items()
        }
        self.results_dirs = {
            k: self.base_dir / "results" / experiment / v
            for k, v in self.run_ids.items()
        }

        # Lazy-loaded models
        self._models: Dict[str, Optional[Network]] = {"epsilon": None, "x_start": None}

        # Load test dataset
        cfg_path = self.model_dirs["epsilon"] / "config.json"
        with open(cfg_path) as f:
            cfg = json.load(f)
        cond = cfg["conditional"]
        data_cfg = cond.get("data", cfg.get("data", {}))
        test_path = data_cfg.get("test_image", "data/flumy_dataset/test")

        # Resolve relative paths against the project root
        if not os.path.isabs(test_path):
            test_path = str(_PROJECT_ROOT / test_path)

        self.image_size = cfg["image_size"]
        self.test_dataset = FlumyDataset(
            data_root=test_path,
            image_size=(self.image_size, self.image_size),
            seed=42,
        )

    # ---- Model loading ----

    def get_model(self, key: str, checkpoint: str = "best_model.pth") -> Network:
        """
        Return a loaded Network for the given scenario, loading lazily.

        Args:
            key: "epsilon" or "x_start"
            checkpoint: Which checkpoint file to load.
        """
        if key not in self.run_ids:
            raise ValueError(f"key must be 'epsilon' or 'x_start', got '{key}'")
        if self._models[key] is None:
            print(
                f"Loading model [{self.labels[key]}] from {self.model_dirs[key] / checkpoint} …"
            )
            self._models[key] = load_model(
                model_dir=str(self.model_dirs[key]),
                checkpoint=checkpoint,
                device=self.device,
            )
        return self._models[key]

    # ---- Loss curves ----

    def read_losses(self) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
        """Read TensorBoard scalars for both runs."""
        losses = {}
        for key in ("epsilon", "x_start"):
            log_dir = self.log_dirs[key]
            if log_dir.is_dir():
                losses[key] = read_tensorboard_logs(str(log_dir))
            else:
                print(f"Warning: log dir not found: {log_dir}")
                losses[key] = {}
        return losses

    def plot_losses(
        self,
        tags: Optional[List[str]] = None,
        smoothing: float = 0.95,
        figsize: Tuple[int, int] = (14, 5),
        save_path: Optional[str] = None,
    ):
        """
        Plot training/validation loss curves for both models side-by-side.

        Args:
            tags: Which scalar tags to plot (default: all Loss/* tags).
            smoothing: Exponential moving average smoothing factor (0=none, 1=max).
            figsize: Figure size.
            save_path: If set, save figure to this path.
        """
        all_losses = self.read_losses()

        # Collect all unique tags
        if tags is None:
            tag_set = set()
            for v in all_losses.values():
                tag_set.update(t for t in v if "Loss" in t or "loss" in t.lower())
            tags = sorted(tag_set) if tag_set else ["Loss/Train"]

        fig, axes = plt.subplots(1, len(tags), figsize=figsize, squeeze=False)
        colors = {"epsilon": "#1f77b4", "x_start": "#ff7f0e"}

        for col, tag in enumerate(tags):
            ax = axes[0, col]
            for key in ("epsilon", "x_start"):
                data = all_losses.get(key, {}).get(tag, [])
                if not data:
                    continue
                steps, values = zip(*data)
                steps = np.array(steps)
                values = np.array(values)

                # Smoothed curve
                smoothed = _ema_smooth(values, smoothing)
                ax.plot(
                    steps,
                    smoothed,
                    color=colors[key],
                    label=self.labels[key],
                    linewidth=1.5,
                )
                ax.plot(steps, values, color=colors[key], alpha=0.15, linewidth=0.5)

            ax.set_xlabel("Step" if "Train" in tag else "Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(tag.replace("/", " – "), fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")

        fig.suptitle("Training Loss Comparison", fontsize=14, y=1.02)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved → {save_path}")
        plt.show()
        return fig

    # ---- Inference ----

    def get_test_sample(self, index: int = 0) -> dict:
        """Return a single test sample dict from the dataset."""
        return self.test_dataset[index]

    def infer(
        self,
        key: str,
        cond_image: torch.Tensor,
        use_ddim: bool = True,
        ddim_steps: int = 50,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Run inference for one model.

        Args:
            key: "epsilon" or "x_start"
            cond_image: (1, 1, H, W) conditioning tensor.
            use_ddim: Use DDIM sampling.
            ddim_steps: DDIM steps.
            seed: Random seed for reproducibility.

        Returns:
            Facies map as int8 array (H, W).
        """
        model = self.get_model(key)
        if seed is not None:
            torch.manual_seed(seed)
        return run_inference(
            model,
            cond_image,
            device=self.device,
            use_ddim=use_ddim,
            ddim_steps=ddim_steps,
        )

    # ---- Comparison plots ----

    def plot_input_data(
        self,
        index: int = 0,
        num_samples: int = 3,
        figsize: Tuple[int, int] = (16, 4),
        save_path: Optional[str] = None,
    ):
        """
        Plot ground truth facies maps and their RMS conditioning inputs.

        Args:
            index: Starting index in the test dataset.
            num_samples: Number of samples to show.
            figsize: Figure size.
            save_path: Save path for the figure.
        """
        n = min(num_samples, len(self.test_dataset) - index)
        fig, axes = plt.subplots(2, n, figsize=figsize)
        if n == 1:
            axes = axes[:, np.newaxis]

        for i in range(n):
            sample = self.get_test_sample(index + i)
            gt = denormalize_facies(sample["gt_image"].squeeze().numpy())
            rms = sample["cond_image"].squeeze().numpy()

            plot_facies(
                axes[0, i], gt, title=f"Facies #{index + i}", show_colorbar=(i == n - 1)
            )
            plot_rms(
                axes[1, i], rms, title=f"RMS #{index + i}", show_colorbar=(i == n - 1)
            )

        axes[0, 0].set_ylabel("Ground Truth\nFacies", fontsize=10)
        axes[1, 0].set_ylabel("RMS Input\n(Conditioning)", fontsize=10)
        fig.suptitle("Input Data Overview", fontsize=14, y=1.02)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved → {save_path}")
        plt.show()
        return fig

    def infer_and_compare(
        self,
        index: int = 0,
        use_ddim: bool = True,
        ddim_steps: int = 50,
        seed: int = 42,
        figsize: Tuple[int, int] = (16, 4),
        save_path: Optional[str] = None,
    ):
        """
        Run inference for both models on a test sample and plot comparison.

        Shows: Ground Truth | Predict-Noise (ε) | Predict-Image (x₀)

        Args:
            index: Test dataset index.
            use_ddim: Use DDIM sampling.
            ddim_steps: DDIM steps.
            seed: Seed for reproducible sampling.
            figsize: Figure size.
            save_path: Save path for the figure.
        """
        sample = self.get_test_sample(index)
        cond_image = sample["cond_image"].unsqueeze(0)  # (1, 1, H, W)
        gt = denormalize_facies(sample["gt_image"].squeeze().numpy())

        pred_eps = self.infer(
            "epsilon", cond_image, use_ddim=use_ddim, ddim_steps=ddim_steps, seed=seed
        )
        pred_x0 = self.infer(
            "x_start", cond_image, use_ddim=use_ddim, ddim_steps=ddim_steps, seed=seed
        )

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        plot_facies(axes[0], gt, title="Ground Truth")
        plot_facies(axes[1], pred_eps, title=self.labels["epsilon"])
        plot_facies(axes[2], pred_x0, title=self.labels["x_start"])

        fig.suptitle(f"Test Sample #{index}", fontsize=14, y=1.02)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved → {save_path}")
        plt.show()
        return fig

    def infer_and_compare_with_rms(
        self,
        index: int = 0,
        use_ddim: bool = True,
        ddim_steps: int = 50,
        seed: int = 42,
        figsize: Tuple[int, int] = (20, 4),
        save_path: Optional[str] = None,
    ):
        """
        Like infer_and_compare but also shows the RMS conditioning input.

        Shows: RMS Input | Ground Truth | Predict-Noise (ε) | Predict-Image (x₀)
        """
        sample = self.get_test_sample(index)
        cond_image = sample["cond_image"].unsqueeze(0)
        gt = denormalize_facies(sample["gt_image"].squeeze().numpy())
        rms = sample["cond_image"].squeeze().numpy()

        pred_eps = self.infer(
            "epsilon", cond_image, use_ddim=use_ddim, ddim_steps=ddim_steps, seed=seed
        )
        pred_x0 = self.infer(
            "x_start", cond_image, use_ddim=use_ddim, ddim_steps=ddim_steps, seed=seed
        )

        fig, axes = plt.subplots(1, 4, figsize=figsize)
        plot_rms(axes[0], rms, title="RMS Input")
        plot_facies(axes[1], gt, title="Ground Truth")
        plot_facies(axes[2], pred_eps, title=self.labels["epsilon"])
        plot_facies(axes[3], pred_x0, title=self.labels["x_start"])

        fig.suptitle(f"Test Sample #{index}", fontsize=14, y=1.02)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved → {save_path}")
        plt.show()
        return fig

    def batch_compare(
        self,
        indices: Optional[List[int]] = None,
        num_samples: int = 5,
        use_ddim: bool = True,
        ddim_steps: int = 50,
        seed: int = 42,
        figsize_per_row: Tuple[float, float] = (20, 3.5),
        save_path: Optional[str] = None,
    ):
        """
        Compare multiple test samples in a grid.

        Layout: each row = one sample with columns [RMS | GT | ε | x₀]

        Args:
            indices: Specific dataset indices, or None for sequential.
            num_samples: Number of rows (ignored if indices given).
            use_ddim: Use DDIM.
            ddim_steps: DDIM steps.
            seed: Random seed.
            figsize_per_row: Figure width and height per row.
            save_path: Save path.
        """
        if indices is None:
            indices = list(range(min(num_samples, len(self.test_dataset))))

        nrows = len(indices)
        fig, axes = plt.subplots(
            nrows,
            4,
            figsize=(figsize_per_row[0], figsize_per_row[1] * nrows),
        )
        if nrows == 1:
            axes = axes[np.newaxis, :]

        col_titles = [
            "RMS Input",
            "Ground Truth",
            self.labels["epsilon"],
            self.labels["x_start"],
        ]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=11)

        for row, idx in enumerate(indices):
            sample = self.get_test_sample(idx)
            cond_image = sample["cond_image"].unsqueeze(0)
            gt = denormalize_facies(sample["gt_image"].squeeze().numpy())
            rms = sample["cond_image"].squeeze().numpy()

            pred_eps = self.infer(
                "epsilon",
                cond_image,
                use_ddim=use_ddim,
                ddim_steps=ddim_steps,
                seed=seed,
            )
            pred_x0 = self.infer(
                "x_start",
                cond_image,
                use_ddim=use_ddim,
                ddim_steps=ddim_steps,
                seed=seed,
            )

            plot_rms(axes[row, 0], rms, title="", show_colorbar=False)
            plot_facies(axes[row, 1], gt, title="", show_colorbar=False)
            plot_facies(axes[row, 2], pred_eps, title="", show_colorbar=False)
            plot_facies(axes[row, 3], pred_x0, title="", show_colorbar=False)

            axes[row, 0].set_ylabel(f"#{idx}", fontsize=10, rotation=0, labelpad=25)

        fig.suptitle("Batch Comparison", fontsize=14, y=1.01)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved → {save_path}")
        plt.show()
        return fig

    def infer_from_rms_file(
        self,
        rms_path: str,
        use_ddim: bool = True,
        ddim_steps: int = 50,
        seed: int = 42,
        figsize: Tuple[int, int] = (16, 4),
        save_path: Optional[str] = None,
    ):
        """
        Run inference from an arbitrary RMS .npy file (not from the test set).

        Args:
            rms_path: Path to a .npy file containing an RMS amplitude map.
            use_ddim: Use DDIM sampling.
            ddim_steps: DDIM steps.
            seed: Random seed.
            figsize: Figure size.
            save_path: Save path.
        """
        from diffsim.data.seismic import normalize_rms

        rms_raw = np.load(rms_path).astype(np.float32)
        if rms_raw.shape != (self.image_size, self.image_size):
            from PIL import Image

            img = Image.fromarray(rms_raw, mode="F")
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            rms_raw = np.array(img, dtype=np.float32)

        rms_norm = normalize_rms(rms_raw, vmin=-1.0, vmax=1.0)
        cond_image = torch.from_numpy(rms_norm[np.newaxis, np.newaxis, :, :]).float()

        pred_eps = self.infer(
            "epsilon", cond_image, use_ddim=use_ddim, ddim_steps=ddim_steps, seed=seed
        )
        pred_x0 = self.infer(
            "x_start", cond_image, use_ddim=use_ddim, ddim_steps=ddim_steps, seed=seed
        )

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        plot_rms(axes[0], rms_raw, title="RMS Input")
        plot_facies(axes[1], pred_eps, title=self.labels["epsilon"])
        plot_facies(axes[2], pred_x0, title=self.labels["x_start"])

        fig.suptitle(f"Inference from {Path(rms_path).name}", fontsize=14, y=1.02)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved → {save_path}")
        plt.show()
        return fig

    def plot_facies_distribution(
        self,
        index: int = 0,
        use_ddim: bool = True,
        ddim_steps: int = 50,
        seed: int = 42,
        figsize: Tuple[int, int] = (12, 4),
        save_path: Optional[str] = None,
    ):
        """
        Bar chart comparing facies proportions: GT vs ε vs x₀.
        """
        sample = self.get_test_sample(index)
        cond_image = sample["cond_image"].unsqueeze(0)
        gt = denormalize_facies(sample["gt_image"].squeeze().numpy())
        pred_eps = self.infer(
            "epsilon", cond_image, use_ddim=use_ddim, ddim_steps=ddim_steps, seed=seed
        )
        pred_x0 = self.infer(
            "x_start", cond_image, use_ddim=use_ddim, ddim_steps=ddim_steps, seed=seed
        )

        def proportions(arr):
            total = arr.size
            return {FACIES_NAMES[c]: np.sum(arr == c) / total * 100 for c in range(3)}

        gt_p = proportions(gt)
        eps_p = proportions(pred_eps)
        x0_p = proportions(pred_x0)

        labels = list(FACIES_NAMES.values())
        x_pos = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(
            x_pos - width,
            [gt_p[l] for l in labels],
            width,
            label="Ground Truth",
            color="#555555",
        )
        ax.bar(
            x_pos,
            [eps_p[l] for l in labels],
            width,
            label=self.labels["epsilon"],
            color="#1f77b4",
        )
        ax.bar(
            x_pos + width,
            [x0_p[l] for l in labels],
            width,
            label=self.labels["x_start"],
            color="#ff7f0e",
        )

        ax.set_ylabel("Proportion (%)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([l.capitalize() for l in labels])
        ax.legend()
        ax.set_title(f"Facies Distribution – Test #{index}", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved → {save_path}")
        plt.show()
        return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ema_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average smoothing."""
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * values[i]
    return smoothed
