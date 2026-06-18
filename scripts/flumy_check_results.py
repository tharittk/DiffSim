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
FACIES_COLORS = {0: "#1D1D1DE9", 1: "#F5851DC8", 2: "#EEEB37F2"}  # bank  # channel  # point_bar
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
    im = ax.imshow(rms, cmap="gist_rainbow_r", origin="upper")
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
        print(f"Data config from model config: {data_cfg}")
        test_path = data_cfg.get("test_image", "data/flumy_dataset/test")

        print(f"Test dataset path from config: {test_path}")

        # Resolve relative paths against the project root
        if not os.path.isabs(test_path):
            test_path = str(_PROJECT_ROOT / test_path)

        print(f"Loading test dataset from: {test_path}")

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


# ===========================================================================
# SLIDE GENERATION UTILITIES
# ===========================================================================
# Functions below are designed to produce presentation-ready figures for each
# slide in the slides_plan.md.  Each returns a matplotlib Figure that can be
# saved with fig.savefig(...).
# ===========================================================================

# 9-facies colormap (Flumy raw codes)
_RAW_FACIES_LABELS = {
    -1: "Background",
    1: "Channel Lag",
    2: "Point Bar (lower)",
    3: "Point Bar (upper)",
    4: "Point Bar (top)",
    5: "Levee (inner)",
    6: "Levee (outer)",
    7: "Crevasse Splay",
    8: "Overbank (proximal)",
    9: "Overbank (distal)",
}
_RAW_FACIES_COLORS_LIST = [
    "#4A4A4A",  # -1 Background (dark grey)
    "#FFD700",  # 1 Channel Lag (gold)
    "#FFA500",  # 2 Point Bar lower (orange)
    "#FF8C00",  # 3 Point Bar upper (dark orange)
    "#FF6600",  # 4 Point Bar top (deep orange)
    "#C8B060",  # 5 Levee inner (khaki/tan)
    "#A09050",  # 6 Levee outer (dark tan)
    "#D4A850",  # 7 Crevasse Splay (sandy brown)
    "#808080",  # 8 Overbank proximal (grey)
    "#606060",  # 9 Overbank distal (dark grey)
]


def load_3d_cube(
    cube_name: str = "ng10_isbx100_seed1",
    data_dir: str = "/mnt/sda_data/tharitt/diffsim/data/flumy3d",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a matched 3D facies/RMS cube pair.

    Args:
        cube_name: Stem name of the .npy file (without extension).
        data_dir: Root directory containing facies/ and rms/ subdirectories.

    Returns:
        (facies_3d, rms_3d): Raw facies cube (int8) and RMS cube (float32).
    """
    data_dir = Path(data_dir)
    facies_3d = np.load(data_dir / "facies" / f"{cube_name}.npy")
    rms_3d = np.load(data_dir / "rms" / f"{cube_name}.npy")
    return facies_3d, rms_3d


def plot_cube_3d(
    cube: np.ndarray,
    cmap="viridis",
    norm=None,
    slices: Optional[Tuple[int, int, int]] = None,
    title: str = "3D Cube",
    figsize: Tuple[int, int] = (10, 8),
    elev: float = 25,
    azim: float = -60,
    alpha: float = 1.0,
    facies_9class: bool = False,
    save_path: Optional[str] = None,
):
    """
    Render a static 3D cube image with three visible faces showing mid-slices.

    Each face of the cube displays the data slice at the given index:
      - Top face (XY): z-slice
      - Front face (XZ): y-slice
      - Right face (YZ): x-slice

    Args:
        cube: 3D array (nx, ny, nz).
        cmap: Colormap name or ListedColormap instance.
        norm: Optional matplotlib norm (e.g. FACIES_NORM).
        slices: (x_idx, y_idx, z_idx) slice positions. Defaults to midpoints.
        title: Figure title.
        figsize: Figure size.
        elev: Elevation viewing angle in degrees.
        azim: Azimuth viewing angle in degrees.
        alpha: Face transparency (1.0 = opaque).
        facies_9class: If True, remap raw Flumy codes {-1, 1..9} to sequential
            indices and use the 9-class colormap/labels automatically (overrides
            cmap and norm).
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    nx, ny, nz = cube.shape
    if slices is None:
        slices = (nx // 2, ny // 2, nz // 2)
    xi, yi, zi = slices

    # Remap raw 9-class Flumy codes to sequential indices
    if facies_9class:
        code_order = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        mapped = np.zeros_like(cube, dtype=np.int8)
        for seq_idx, code in enumerate(code_order):
            mapped[cube == code] = seq_idx
        cube = mapped
        cmap = mcolors.ListedColormap(_RAW_FACIES_COLORS_LIST)
        norm = mcolors.BoundaryNorm(np.arange(-0.5, 10.5, 1), cmap.N)

    # Resolve colormap
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap
    if norm is None:
        norm = Normalize(vmin=np.nanmin(cube), vmax=np.nanmax(cube))

    def _data_to_rgba(data_2d):
        return cmap_obj(norm(data_2d.astype(float)))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # --- Top face (XY plane at z=zi) ---
    xy_slice = cube[:, :, zi]
    xs = np.arange(nx + 1)
    ys = np.arange(ny + 1)
    Xs, Ys = np.meshgrid(xs, ys, indexing="ij")
    Zs = np.full_like(Xs, float(zi), dtype=float)
    colors_top = _data_to_rgba(xy_slice)
    ax.plot_surface(
        Xs,
        Ys,
        Zs,
        facecolors=colors_top,
        shade=False,
        alpha=alpha,
        rstride=1,
        cstride=1,
    )

    # --- Front face (XZ plane at y=0) ---
    xz_slice = cube[:, 0, :]
    xs = np.arange(nx + 1)
    zs = np.arange(nz + 1)
    Xs_f, Zs_f = np.meshgrid(xs, zs, indexing="ij")
    Ys_f = np.full_like(Xs_f, 0.0, dtype=float)
    colors_front = _data_to_rgba(xz_slice)
    ax.plot_surface(
        Xs_f,
        Ys_f,
        Zs_f,
        facecolors=colors_front,
        shade=False,
        alpha=alpha,
        rstride=1,
        cstride=1,
    )

    # --- Right face (YZ plane at x=nx-1) ---
    yz_slice = cube[-1, :, :]
    ys = np.arange(ny + 1)
    zs = np.arange(nz + 1)
    Ys_r, Zs_r = np.meshgrid(ys, zs, indexing="ij")
    Xs_r = np.full_like(Ys_r, float(nx), dtype=float)
    colors_right = _data_to_rgba(yz_slice)
    ax.plot_surface(
        Xs_r,
        Ys_r,
        Zs_r,
        facecolors=colors_right,
        shade=False,
        alpha=alpha,
        rstride=1,
        cstride=1,
    )

    # --- Mid-slice cross-sections (cut faces) ---
    # Y-slice (front cut at y=yi)
    xz_mid = cube[:, yi, :]
    Xs_m, Zs_m = np.meshgrid(np.arange(nx + 1), np.arange(nz + 1), indexing="ij")
    Ys_m = np.full_like(Xs_m, float(yi), dtype=float)
    colors_ymid = _data_to_rgba(xz_mid)
    ax.plot_surface(
        Xs_m,
        Ys_m,
        Zs_m,
        facecolors=colors_ymid,
        shade=False,
        alpha=alpha,
        rstride=1,
        cstride=1,
    )

    # X-slice (side cut at x=xi)
    yz_mid = cube[xi, :, :]
    Ys_s, Zs_s = np.meshgrid(np.arange(ny + 1), np.arange(nz + 1), indexing="ij")
    Xs_s = np.full_like(Ys_s, float(xi), dtype=float)
    colors_xmid = _data_to_rgba(yz_mid)
    ax.plot_surface(
        Xs_s,
        Ys_s,
        Zs_s,
        facecolors=colors_xmid,
        shade=False,
        alpha=alpha,
        rstride=1,
        cstride=1,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=13, pad=15)

    # Add a colorbar via a ScalarMappable
    import matplotlib.cm as mcm

    sm = mcm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    if facies_9class:
        code_order = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        cbar = plt.colorbar(
            sm, ax=ax, fraction=0.03, pad=0.1, shrink=0.6, ticks=range(10)
        )
        cbar.ax.set_yticklabels([_RAW_FACIES_LABELS[c] for c in code_order], fontsize=7)
    else:
        plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.1, shrink=0.6)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
    return fig


def plot_3d_facies_9class(
    facies_3d: np.ndarray,
    z_index: int = 5,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
):
    """
    Slide 1a: Show a z-slice of the 3D facies cube with all 9 original Flumy facies.

    Args:
        facies_3d: Raw 3D facies array with codes {-1, 1..9}.
        z_index: Depth slice index.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    slice_2d = facies_3d[:, :, z_index]

    # Map raw codes (-1, 1..9) to sequential indices (0..9) for colormap
    code_order = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    mapped = np.zeros_like(slice_2d, dtype=np.int8)
    for seq_idx, code in enumerate(code_order):
        mapped[slice_2d == code] = seq_idx

    cmap = mcolors.ListedColormap(_RAW_FACIES_COLORS_LIST)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, 10.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mapped, cmap=cmap, norm=norm, origin="upper")
    ax.set_title(f"Original 9-Facies Classification (z={z_index})", fontsize=13)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    cbar = plt.colorbar(im, ax=ax, ticks=range(10), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([_RAW_FACIES_LABELS[c] for c in code_order], fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
    return fig


def plot_3d_facies_reclassification(
    facies_3d: np.ndarray,
    z_index: int = 5,
    figsize: Tuple[int, int] = (16, 5),
    save_path: Optional[str] = None,
):
    """
    Slide 1b: Side-by-side comparison of 9-class and 3-class facies.

    Args:
        facies_3d: Raw 3D facies array with codes {-1, 1..9}.
        z_index: Depth slice index.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    from diffsim.data.flumy_generator import FlumyGenerator

    slice_raw = facies_3d[:, :, z_index]
    facies_3class = FlumyGenerator.reclassify_to_three_facies(facies_3d)
    slice_3class = facies_3class[:, :, z_index]

    # Map raw codes
    code_order = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    mapped_raw = np.zeros_like(slice_raw, dtype=np.int8)
    for seq_idx, code in enumerate(code_order):
        mapped_raw[slice_raw == code] = seq_idx

    cmap_raw = mcolors.ListedColormap(_RAW_FACIES_COLORS_LIST)
    norm_raw = mcolors.BoundaryNorm(np.arange(-0.5, 10.5, 1), cmap_raw.N)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: 9-class
    im0 = axes[0].imshow(mapped_raw, cmap=cmap_raw, norm=norm_raw, origin="upper")
    axes[0].set_title("Original (9 facies)", fontsize=12)
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    cbar0 = plt.colorbar(im0, ax=axes[0], ticks=range(10), fraction=0.046, pad=0.04)
    cbar0.ax.set_yticklabels([_RAW_FACIES_LABELS[c] for c in code_order], fontsize=7)

    # Right: 3-class
    im1 = axes[1].imshow(
        slice_3class, cmap=FACIES_CMAP, norm=FACIES_NORM, origin="upper"
    )
    axes[1].set_title("Reclassified (3 facies)", fontsize=12)
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    cbar1 = plt.colorbar(im1, ax=axes[1], ticks=[0, 1, 2], fraction=0.046, pad=0.04)
    cbar1.ax.set_yticklabels(["Mud", "Bank", "Sand"], fontsize=9)

    fig.suptitle(f"Facies Reclassification (z={z_index})", fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
    return fig


def plot_facies_to_rms_pipeline(
    facies_3d: np.ndarray,
    rms_3d: np.ndarray,
    z_index: int = 5,
    figsize: Tuple[int, int] = (22, 4),
    save_path: Optional[str] = None,
):
    """
    Slide 2: Show how the 3D facies cube is converted to a 3D RMS cube.

    Displays: 3-class Facies → AI → Reflectivity → Synthetic Seismic → RMS

    Args:
        facies_3d: Raw 3D facies array (will be reclassified internally).
        rms_3d: Pre-computed RMS cube.
        z_index: Depth slice index.
        figsize: Figure size.
        save_path: Optional save path.
    """
    from diffsim.data.flumy_generator import FlumyGenerator
    from diffsim.data.seismic import (
        facies_to_ai,
        DEFAULT_ROCK_PROPERTIES,
        compute_reflectivity_vertical,
        compute_synthetic_seismic_3d,
        compute_rms_cube,
    )

    facies_reclass = FlumyGenerator.reclassify_to_three_facies(facies_3d)
    facies_slice = facies_reclass[:, :, z_index]

    # Compute AI for the full 3D cube so we can derive reflectivity & synthetic
    rng = np.random.default_rng(42)
    nx, ny, nz = facies_reclass.shape
    ai_3d = np.empty((nx, ny, nz), dtype=np.float32)
    for zi in range(nz):
        ai_3d[:, :, zi] = facies_to_ai(
            facies_reclass[:, :, zi],
            rock_properties=DEFAULT_ROCK_PROPERTIES,
            rng=rng,
        )

    ai_slice = ai_3d[:, :, z_index]

    # Reflectivity & synthetic seismic via proper seismic forward modeling
    reflectivity_3d = compute_reflectivity_vertical(ai_3d)
    synthetic_3d = compute_synthetic_seismic_3d(reflectivity_3d)

    # Map z_index into reflectivity/synthetic z-axis (nz-1 layers)
    refl_z = min(z_index, reflectivity_3d.shape[2] - 1)
    reflectivity_slice = reflectivity_3d[:, :, refl_z]
    synthetic_slice = synthetic_3d[:, :, refl_z]

    # RMS slice (pre-computed)
    rms_z = min(z_index, rms_3d.shape[2] - 1)
    rms_slice = rms_3d[:, :, rms_z]

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Facies
    im0 = axes[0].imshow(
        facies_slice, cmap=FACIES_CMAP, norm=FACIES_NORM, origin="upper"
    )
    axes[0].set_title("① Facies (3-class)", fontsize=10)
    plt.colorbar(
        im0, ax=axes[0], ticks=[0, 1, 2], fraction=0.046, pad=0.04
    ).ax.set_yticklabels(["Mud", "Bank", "Sand"], fontsize=7)

    # AI
    im1 = axes[1].imshow(ai_slice, cmap="gray_r", origin="upper")
    axes[1].set_title("② Acoustic Impedance", fontsize=10)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # # Reflectivity
    # im2 = axes[2].imshow(reflectivity_slice, cmap="gray_r", origin="upper")
    # axes[2].set_title("③ Reflectivity", fontsize=10)
    # plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Synthetic seismic/
    im3 = axes[2].imshow(synthetic_slice, cmap="gray_r", origin="upper")
    axes[2].set_title("③ Synthetic Seismic\n(Reflectivity ⊛ Ricker)", fontsize=10)
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # RMS
    im4 = axes[3].imshow(rms_slice, cmap="gist_rainbow_r", origin="upper")
    axes[3].set_title("④ RMS Amplitude", fontsize=10)
    plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Seismic Forward Modeling Pipeline", fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
    return fig


def plot_slicing_and_cropping(
    facies_3d: np.ndarray,
    rms_3d: np.ndarray,
    z_index: int = 5,
    crop_size: int = 64,
    num_crops: int = 3,
    seed: int = 42,
    figsize: Tuple[int, int] = (18, 8),
    save_path: Optional[str] = None,
):
    """
    Slide 3: Show how 3D cubes are sliced at depth z and randomly cropped to 64x64.

    Top row: full 256x256 slice with crop rectangles drawn.
    Bottom row: the cropped 64x64 patches (facies + RMS pairs).

    Args:
        facies_3d: Raw 3D facies array.
        rms_3d: 3D RMS cube.
        z_index: Depth slice.
        crop_size: Crop size (64).
        num_crops: Number of random crops to show.
        seed: Random seed.
        figsize: Figure size.
        save_path: Optional save path.
    """
    from diffsim.data.flumy_generator import FlumyGenerator
    import matplotlib.patches as mpatches

    facies_reclass = FlumyGenerator.reclassify_to_three_facies(facies_3d)
    facies_slice = facies_reclass[:, :, z_index]
    rms_z = min(z_index, rms_3d.shape[2] - 1)
    rms_slice = rms_3d[:, :, rms_z]

    rng = np.random.default_rng(seed)
    h, w = facies_slice.shape
    crops = []
    for _ in range(num_crops):
        y = rng.integers(0, h - crop_size + 1)
        x = rng.integers(0, w - crop_size + 1)
        crops.append((y, x))

    rect_colors = plt.cm.Set1(np.linspace(0, 1, num_crops))

    # Use the full slice's color range for all RMS panels
    rms_vmin, rms_vmax = float(rms_slice.min()), float(rms_slice.max())

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, num_crops + 1, width_ratios=[2] + [1] * num_crops)

    # Top-left: full facies slice with crop boxes
    ax_full_f = fig.add_subplot(gs[0, 0])
    ax_full_f.imshow(facies_slice, cmap=FACIES_CMAP, norm=FACIES_NORM, origin="upper")
    ax_full_f.set_title(f"Full Facies Slice (z={z_index})\n256×256", fontsize=10)
    ax_full_f.set_xticks([])
    ax_full_f.set_yticks([])

    # Top-right equivalent: full RMS slice with crop boxes
    ax_full_r = fig.add_subplot(gs[1, 0])
    ax_full_r.imshow(
        rms_slice, cmap="gist_rainbow_r", vmin=rms_vmin, vmax=rms_vmax, origin="upper"
    )
    ax_full_r.set_title(f"Full RMS Slice (z={z_index})\n256×256", fontsize=10)
    ax_full_r.set_xticks([])
    ax_full_r.set_yticks([])

    for i, (y, x) in enumerate(crops):
        color = rect_colors[i]
        for ax_full in [ax_full_f, ax_full_r]:
            rect = mpatches.Rectangle(
                (x, y),
                crop_size,
                crop_size,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            ax_full.add_patch(rect)

        # Cropped facies
        ax_cf = fig.add_subplot(gs[0, i + 1])
        ax_cf.imshow(
            facies_slice[y : y + crop_size, x : x + crop_size],
            cmap=FACIES_CMAP,
            norm=FACIES_NORM,
            origin="upper",
        )
        ax_cf.set_title(f"Crop {i + 1}\n{crop_size}×{crop_size}", fontsize=9)
        ax_cf.set_xticks([])
        ax_cf.set_yticks([])
        for spine in ax_cf.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

        # Cropped RMS
        ax_cr = fig.add_subplot(gs[1, i + 1])
        ax_cr.imshow(
            rms_slice[y : y + crop_size, x : x + crop_size],
            cmap="gist_rainbow_r",
            vmin=rms_vmin,
            vmax=rms_vmax,
            origin="upper",
        )
        ax_cr.set_title(f"Crop {i + 1}\n{crop_size}×{crop_size}", fontsize=9)
        ax_cr.set_xticks([])
        ax_cr.set_yticks([])
        for spine in ax_cr.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    fig.suptitle(
        "Slicing & Random Cropping: 3D Cube → 2D Training Patches", fontsize=14, y=1.02
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
    return fig


def plot_diffusion_training_schematic(
    facies_patch: Optional[np.ndarray] = None,
    rms_patch: Optional[np.ndarray] = None,
    num_noise_levels: int = 5,
    figsize: Tuple[int, int] = (18, 6),
    save_path: Optional[str] = None,
):
    """
    Slide 4: Show how 2D facies patches are fed to the diffusion network.

    Illustrates the forward diffusion (adding noise) and the reverse process.
    Top row: progressively noised facies (t=0 → T).
    Bottom row: RMS conditioning (constant) + arrow → UNet → denoised output.

    Args:
        facies_patch: (64, 64) int8 facies. If None, loads from test dataset.
        rms_patch: (64, 64) float32 RMS. If None, loads from test dataset.
        num_noise_levels: Number of intermediate noise steps to show.
        figsize: Figure size.
        save_path: Optional save path.
    """
    from diffsim.data.flumy_generator import normalize_facies
    from diffsim.data.seismic import normalize_rms

    # Load a sample if not provided
    if facies_patch is None or rms_patch is None:
        test_dir = _PROJECT_ROOT / "data" / "flumy_dataset" / "test"
        facies_files = sorted((test_dir / "facies").glob("*.npy"))
        rms_files = sorted((test_dir / "rms").glob("*.npy"))
        facies_patch = np.load(facies_files[0])
        rms_patch = np.load(rms_files[0])

    facies_norm = normalize_facies(facies_patch).astype(np.float32)
    rms_norm = normalize_rms(rms_patch, vmin=-1.0, vmax=1.0)

    # Simulate forward diffusion (add noise at various levels)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(facies_norm.shape).astype(np.float32)
    t_fracs = np.linspace(0, 1, num_noise_levels)
    noisy_images = []
    for t in t_fracs:
        alpha = 1.0 - t  # simplified schedule
        noisy = np.sqrt(alpha) * facies_norm + np.sqrt(1 - alpha) * noise
        noisy_images.append(noisy)

    ncols = num_noise_levels + 1  # +1 for RMS conditioning
    fig, axes = plt.subplots(2, ncols, figsize=figsize)

    # Top row: forward diffusion (progressive noise)
    axes[0, 0].set_ylabel("Forward\nDiffusion\n(add noise)", fontsize=9)
    for i, (t, img) in enumerate(zip(t_fracs, noisy_images)):
        axes[0, i].imshow(img, cmap="gray", vmin=-1.5, vmax=1.5, origin="upper")
        if i == 0:
            axes[0, i].set_title("x₀ (clean)", fontsize=9)
        elif i == num_noise_levels - 1:
            axes[0, i].set_title(f"x_T (pure noise)", fontsize=9)
        else:
            axes[0, i].set_title(f"t={t:.1f}", fontsize=9)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
    # Last col top: empty (placeholder for arrow)
    axes[0, -1].axis("off")
    axes[0, -1].text(
        0.5,
        0.5,
        "→ UNet →\nDenoises",
        ha="center",
        va="center",
        fontsize=11,
        style="italic",
        transform=axes[0, -1].transAxes,
    )

    # Bottom row: RMS conditioning + reverse outputs
    axes[1, 0].imshow(rms_norm, cmap="gist_rainbow_r", origin="upper")
    axes[1, 0].set_title("RMS Conditioning\n(input to UNet)", fontsize=9)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    axes[1, 0].set_ylabel("Reverse\nProcess\n(denoise)", fontsize=9)

    # Simulate reverse (just show decreasing noise for illustration)
    for i in range(1, num_noise_levels):
        t_rev = t_fracs[num_noise_levels - 1 - i]
        alpha = 1.0 - t_rev
        denoised = np.sqrt(alpha) * facies_norm + np.sqrt(1 - alpha) * noise * 0.3
        axes[1, i].imshow(denoised, cmap="gray", vmin=-1.5, vmax=1.5, origin="upper")
        axes[1, i].set_title(f"Denoise step {i}", fontsize=9)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    # Final output
    axes[1, -1].imshow(facies_patch, cmap=FACIES_CMAP, norm=FACIES_NORM, origin="upper")
    axes[1, -1].set_title("Output Facies\n(classified)", fontsize=9)
    axes[1, -1].set_xticks([])
    axes[1, -1].set_yticks([])

    fig.suptitle("Conditional Diffusion Training: RMS → Facies", fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
    return fig


def plot_3d_cube_orthogonal(
    cube: np.ndarray,
    cmap="viridis",
    norm=None,
    slices: Optional[Tuple[int, int, int]] = None,
    title: str = "3D Cube",
    figsize: Tuple[int, int] = (14, 4),
    save_path: Optional[str] = None,
):
    """
    Show 3 orthogonal slices (XY, XZ, YZ) through a 3D volume.

    Useful for visualizing the full 3D structure of facies or RMS cubes.

    Args:
        cube: 3D array (nx, ny, nz).
        cmap: Colormap name or instance.
        norm: Optional norm (e.g. FACIES_NORM for facies).
        slices: (x_idx, y_idx, z_idx) slice positions. Defaults to midpoints.
        title: Figure title.
        figsize: Figure size.
        save_path: Optional save path.
    """
    nx, ny, nz = cube.shape
    if slices is None:
        slices = (nx // 2, ny // 2, nz // 2)
    xi, yi, zi = slices

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # XY slice (plan view)
    axes[0].imshow(cube[:, :, zi], cmap=cmap, norm=norm, origin="upper", aspect="equal")
    axes[0].set_title(f"Plan view (z={zi})", fontsize=10)
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    # XZ slice (cross-section along Y=yi)
    axes[1].imshow(
        cube[:, yi, :].T, cmap=cmap, norm=norm, origin="upper", aspect="auto"
    )
    axes[1].set_title(f"Cross-section (y={yi})", fontsize=10)
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Z")

    # YZ slice (cross-section along X=xi)
    axes[2].imshow(
        cube[xi, :, :].T, cmap=cmap, norm=norm, origin="upper", aspect="auto"
    )
    axes[2].set_title(f"Cross-section (x={xi})", fontsize=10)
    axes[2].set_xlabel("Y")
    axes[2].set_ylabel("Z")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
    return fig


def plot_predict_type_explanation(
    figsize: Tuple[int, int] = (16, 5),
    save_path: Optional[str] = None,
):
    """
    Slide 6 supporting: Diagram explaining the two predict_type approaches.

    Left: predict ε (noise) then subtract to get x₀.
    Right: predict x₀ directly.

    Returns a schematic figure (no data needed).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Epsilon prediction schematic
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Scenario A: Predict Noise (ε)", fontsize=12, fontweight="bold", color="#1f77b4"
    )

    # Boxes
    boxes = [
        (0.5, 4, "x_t\n(noisy)"),
        (4, 4, "UNet"),
        (7.5, 4, "ε̂\n(predicted\nnoise)"),
        (7.5, 1, "x̂₀ = x_t − √(1−ᾱ)·ε̂\n/ √ᾱ"),
    ]
    for x, y, txt in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, y - 0.8),
                2,
                1.6,
                fill=True,
                facecolor="#e3f2fd",
                edgecolor="#1f77b4",
                linewidth=2,
            )
        )
        ax.text(x + 1, y, txt, ha="center", va="center", fontsize=8)

    ax.annotate(
        "",
        xy=(3.9, 4),
        xytext=(2.6, 4),
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2),
    )
    ax.annotate(
        "",
        xy=(7.4, 4),
        xytext=(6.1, 4),
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2),
    )
    ax.annotate(
        "",
        xy=(8.5, 3.1),
        xytext=(8.5, 1.9),
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2),
    )

    # X_start prediction schematic
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Scenario B: Predict Image (x₀)",
        fontsize=12,
        fontweight="bold",
        color="#ff7f0e",
    )

    boxes = [
        (0.5, 4, "x_t\n(noisy)"),
        (4, 4, "UNet"),
        (7.5, 4, "x̂₀\n(predicted\nimage)"),
    ]
    for x, y, txt in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, y - 0.8),
                2,
                1.6,
                fill=True,
                facecolor="#fff3e0",
                edgecolor="#ff7f0e",
                linewidth=2,
            )
        )
        ax.text(x + 1, y, txt, ha="center", va="center", fontsize=8)

    ax.annotate(
        "",
        xy=(3.9, 4),
        xytext=(2.6, 4),
        arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=2),
    )
    ax.annotate(
        "",
        xy=(7.4, 4),
        xytext=(6.1, 4),
        arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=2),
    )

    # RMS conditioning note (both)
    for i, ax in enumerate(axes):
        ax.text(
            5,
            5.5,
            "RMS conditioning concatenated to input",
            ha="center",
            va="center",
            fontsize=8,
            style="italic",
            color="gray",
        )

    fig.suptitle(
        "Two Prediction Strategies for Conditional Diffusion", fontsize=14, y=1.02
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
    return fig
