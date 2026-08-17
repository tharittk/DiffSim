#!/usr/bin/env python
"""
Unified conditional diffusion training script with multi-GPU support.

Usage:
    # Single GPU
    python train_conditional.py --config configs/case1_geomodeling.json

    # Multi-GPU (e.g., 4 GPUs)
    python train_conditional.py --config configs/case1_geomodeling.json -gpu 0,1,2,3

Supports both 2D and 3D cases based on config["type"].
"""

import json
import os
import re
import socket
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])


from torch.utils.data import DataLoader, Subset, WeightedRandomSampler  # noqa: E402
from torch.optim import Adam, lr_scheduler
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from tqdm import tqdm

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffsim.core.network import Network
from diffsim.models.guided_diffusion import UNet, UNet3D
from diffsim.data import InpaintDatasetCase1, InpaintDatasetCase2, NPYInpaintDataset
from diffsim.data.flumy_dataset import FlumyDataset


def set_seed(seed, deterministic=True):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def configure_cuda_performance(enable_tf32, cudnn_benchmark):
    """Configure CUDA execution without changing model architecture."""
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32
    if enable_tf32:
        torch.set_float32_matmul_precision("high")


def compute_class_pixel_weights(dataset, num_classes=3, sample_size=4000, power=1.0, seed=42):
    """Estimate inverse-frequency facies class weights from a random sample.

    Reads only facies files (not RMS) from a random subset of the dataset to
    keep this cheap. Weights are normalized to a mean of 1 so overall loss
    magnitude stays comparable to the unweighted case.
    """
    rng = np.random.default_rng(seed)
    facies_paths = [facies_path for facies_path, _ in dataset.file_pairs]
    if sample_size < len(facies_paths):
        indices = rng.choice(len(facies_paths), size=sample_size, replace=False)
        facies_paths = [facies_paths[i] for i in indices]

    counts = np.zeros(num_classes, dtype=np.int64)
    for facies_path in facies_paths:
        arr = np.load(facies_path)
        counts += np.bincount(arr.ravel().astype(np.int64), minlength=num_classes)[:num_classes]

    freq = counts / counts.sum()
    weights = (1.0 / freq) ** float(power)
    weights *= num_classes / weights.sum()
    return torch.from_numpy(weights.astype(np.float32))


def build_group_balanced_sampler(dataset, group_power=1.0):
    """Build a WeightedRandomSampler using (ng, isbx) parsed from filenames.

    Weight for sample i in group g is proportional to 1 / count(g)^group_power.
    This equalizes sampling across scenarios without scanning array contents.
    """
    pat = re.compile(r"ng(?P<ng>\d+)_isbx(?P<isbx>\d+)_")
    group_keys = []
    for facies_path, _ in dataset.file_pairs:
        m = pat.search(facies_path.name)
        if m:
            group_keys.append((int(m.group("ng")), int(m.group("isbx"))))
        else:
            group_keys.append((None, None))

    # Count samples per group
    group_counts = {}
    for key in group_keys:
        group_counts[key] = group_counts.get(key, 0) + 1

    # Inverse-frequency weighting (optionally tempered)
    weights = np.array(
        [1.0 / (group_counts[key] ** float(group_power)) for key in group_keys],
        dtype=np.float64,
    )

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(weights),
        replacement=True,
    )
    return sampler, group_counts


def main_worker(config_path, timestamp):
    """Worker function for each GPU process."""

    # Setup distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Extract conditional config
    cond = config["conditional"]
    train_cfg = cond["training"]
    set_seed(42, deterministic=train_cfg.get("deterministic", True))
    configure_cuda_performance(
        enable_tf32=train_cfg.get("enable_tf32", False),
        cudnn_benchmark=train_cfg.get("cudnn_benchmark", False),
    )

    # Build UNet config
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

    # Choose module type and predict_type
    module_name = cond.get("module_name", "guided_diffusion")
    predict_type = cond.get("predict_type", "epsilon")

    # Build network
    network = Network(
        unet=unet_config,
        beta_schedule=cond["beta_schedule"],
        module_name=module_name,
        predict_type=predict_type,
    )
    network.to(device)
    net_module = network

    # Setup noise schedule
    net_module.set_new_noise_schedule(device=device, phase="train")

    # Set loss function
    if train_cfg["loss"] == "mse":
        loss_fn = nn.MSELoss()
    elif train_cfg["loss"] == "l1":
        loss_fn = nn.L1Loss()
    else:
        loss_fn = nn.MSELoss()
    net_module.set_loss(loss_fn)

    # Setup output directories (only rank 0 creates)
    exp_name = config["name"]
    output_cfg = cond.get("output", {})
    base_results_dir = Path(
        output_cfg.get("results_dir", f"./results/{exp_name}_conditional")
    )
    base_model_dir = Path(
        output_cfg.get("model_dir", f"./model/{exp_name}_conditional")
    )
    base_log_dir = Path(output_cfg.get("log_dir", str(base_results_dir / "logs")))

    results_dir = base_results_dir / timestamp
    model_dir = base_model_dir / timestamp
    log_dir = base_log_dir / timestamp

    # make directories if needed
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # Setup dataset
    data_cfg = cond.get("data", config.get("data", {}))
    data_path = data_cfg.get(
        "train_image", data_cfg.get("train_image_path", data_cfg.get("train_path", ""))
    )

    image_size = config["image_size"]
    num_workers = train_cfg.get("num_workers", 4)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=train_cfg.get("persistent_workers", False),
            prefetch_factor=train_cfg.get("prefetch_factor", 2),
        )

    # Flumy dataset: loads paired facies/rms files from split subdirectories.
    train_data_path = data_path
    dataset = FlumyDataset(
        data_root=train_data_path,
        image_size=(image_size, image_size),
    )

    # Optional: balance scenario sampling by (ng, isbx) from filename metadata.
    # This helps prevent degenerate predictions on low/high-end combinations.
    balance_by_ng_isbx = train_cfg.get("balance_by_ng_isbx", True)
    group_power = train_cfg.get("balance_group_power", 1.0)
    if balance_by_ng_isbx:
        sampler, group_counts = build_group_balanced_sampler(
            dataset, group_power=group_power
        )
        dataloader = DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            sampler=sampler,
            shuffle=False,
            **loader_kwargs,
        )
        print(
            f"Using weighted sampler by (ng,isbx): {len(group_counts)} groups, "
            f"power={group_power}"
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            **loader_kwargs,
        )

    print(f"Data path: {data_path}")
    print(f"Dataset class: {dataset.__class__.__name__}")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"DataLoader workers: {num_workers}")

    # Optional: class-balanced pixel loss to counter facies imbalance across
    # scenarios (e.g. low ng is shale-dominant, high ng is sand-dominant).
    class_weights = None
    if train_cfg.get("class_balanced_loss", False):
        class_weights = compute_class_pixel_weights(
            dataset,
            sample_size=train_cfg.get("class_weight_sample_size", 4000),
            power=train_cfg.get("class_weight_power", 1.0),
        ).to(device)
        print(f"Class-balanced loss enabled. Class weights: {class_weights.tolist()}")

    # Setup optimizer and scheduler
    optimizer = Adam(network.parameters(), lr=train_cfg["lr"])
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=10, factor=0.5
    )

    # Gradient clipping (prevents training instability)
    grad_clip = train_cfg.get("grad_clip", 1.0)

    # Optional: EMA (only on main process)
    ema_decay = train_cfg.get("ema_decay", None)
    if ema_decay:
        ema_state = {k: v.clone() for k, v in net_module.state_dict().items()}

    # Save config for reproducibility
    import shutil

    shutil.copy2(config_path, str(model_dir / "config.json"))

    # Setup validation dataset if test data path is available
    val_dataloader = None
    val_ng_loaders = {}
    test_data_path = data_cfg.get("test_image", None)
    if test_data_path and os.path.isdir(test_data_path):
        val_dataset = FlumyDataset(
            data_root=test_data_path,
            image_size=(image_size, image_size),
            seed=123,  # fixed seed for consistent validation
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            **loader_kwargs,
        )

        # Stratify validation by ng tertile so scenario-specific regressions
        # (e.g. low net-to-gross collapsing to the wrong majority facies) are
        # visible without manual QC notebook spot-checks.
        ng_pat = re.compile(r"ng(?P<ng>\d+)_")
        bucket_indices = {"low_ng": [], "mid_ng": [], "high_ng": []}
        for idx, (facies_path, _) in enumerate(val_dataset.file_pairs):
            m = ng_pat.search(facies_path.name)
            if not m:
                continue
            ng = int(m.group("ng"))
            if ng <= 30:
                bucket_indices["low_ng"].append(idx)
            elif ng <= 60:
                bucket_indices["mid_ng"].append(idx)
            else:
                bucket_indices["high_ng"].append(idx)
        val_ng_loaders = {
            name: DataLoader(
                Subset(val_dataset, idxs),
                batch_size=train_cfg["batch_size"],
                shuffle=False,
                **loader_kwargs,
            )
            for name, idxs in bucket_indices.items()
            if idxs
        }

    use_amp = device.type == "cuda" and train_cfg.get("amp", False)
    amp_dtype_name = train_cfg.get("amp_dtype", "bfloat16")
    if amp_dtype_name != "bfloat16":
        raise ValueError(
            f"Unsupported amp_dtype={amp_dtype_name!r}; use 'bfloat16'."
        )
    amp_dtype = torch.bfloat16
    print(f"AMP: {'bfloat16' if use_amp else 'disabled'}")

    # Resume from checkpoint if specified
    start_epoch = 0
    # resume_path = train_cfg.get("resume_checkpoint", None)
    # if resume_path and os.path.isfile(resume_path):
    #     print(f"Resuming from checkpoint: {resume_path}")
    #     checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    #     if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    #         net_module.load_state_dict(checkpoint["model_state"])
    #         if "optimizer_state" in checkpoint:
    #             optimizer.load_state_dict(checkpoint["optimizer_state"])
    #         if "epoch" in checkpoint:
    #             start_epoch = checkpoint["epoch"] + 1
    #         if ema_decay and "ema_state" in checkpoint:
    #             ema_state = checkpoint["ema_state"]
    #     else:
    #         # Legacy format: just state dict
    #         net_module.load_state_dict(checkpoint)

    # Training loop
    epochs = train_cfg["epochs"]
    save_every = train_cfg.get("save_every", 500)
    checkpoint_every = train_cfg.get("checkpoint_every", 10)
    min_loss = float("inf")
    best_epoch = 0

    print(f"Starting conditional training for {exp_name}")
    print(f"Run ID: {timestamp}")
    print(f"Model type: {'2D'}, Predict type: {predict_type}")
    print(f"Epochs: {epochs}, Batch size: {train_cfg['batch_size']}")
    print(
        f"Save samples every: {save_every} steps, Checkpoint every: {checkpoint_every} epochs"
    )

    for epoch in range(start_epoch, epochs):
        # Set epoch for sampler (important for shuffling)

        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(pbar):
            optimizer.zero_grad(set_to_none=True)

            # Get batch data
            gt_image = batch["gt_image"].to(device, non_blocking=True)
            cond_image = batch["cond_image"]
            if isinstance(cond_image, np.ndarray):
                cond_image = torch.from_numpy(cond_image).float()
            cond_image = cond_image.to(device, non_blocking=True)

            # there is no well locations
            # mask = batch["mask"].to(device)
            mask = None

            pixel_weight = None
            if class_weights is not None:
                # LITHO_NORMALIZED maps shale/sand/silt to -1/0/1.
                class_idx = torch.round(gt_image + 1.0).long().clamp(0, len(class_weights) - 1)
                pixel_weight = class_weights[class_idx]

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                loss = network(gt_image, y_cond=cond_image, mask=mask, pixel_weight=pixel_weight)

            epoch_losses.append(loss.item())

            # Backprop
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
            optimizer.step()

            # Update EMA (only on main)
            if ema_decay:
                with torch.no_grad():
                    for k, v in net_module.state_dict().items():
                        ema_state[k] = ema_decay * ema_state[k] + (1 - ema_decay) * v

            # Logging (only on main)
            global_step = epoch * len(dataloader) + step
            writer.add_scalar("Loss/Train", loss.item(), global_step)
            writer.add_scalar(
                "Learning_rate", optimizer.param_groups[0]["lr"], global_step
            )
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

            # Save samples periodically (only on main)
            if global_step != 0 and global_step % save_every == 0:
                net_module.eval()
                with torch.no_grad():
                    sample_cond = cond_image[:4]
                    sample_gt = gt_image[:4]
                    # sample_mask = mask[:4]
                    sample_mask = None

                    output, _ = net_module.restoration(
                        sample_cond, y_0=sample_gt, mask=sample_mask
                    )

                    save_image(
                        output,
                        str(results_dir / f"output_{epoch}_{global_step}.png"),
                        nrow=2,
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    save_image(
                        sample_gt,
                        str(results_dir / f"gt_{epoch}_{global_step}.png"),
                        nrow=2,
                        normalize=True,
                        value_range=(-1, 1),
                    )
                net_module.train()

        # Sync losses across processes
        avg_loss = np.mean(epoch_losses)

        # Always evaluate and checkpoint the final epoch, even when it is not
        # an exact multiple of the configured interval.
        is_checkpoint_epoch = (
            epoch % checkpoint_every == 0 or epoch == epochs - 1
        )

        # Save checkpoint (with full state for resumability)
        if is_checkpoint_epoch:
            checkpoint_state = {
                "model_state": net_module.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "min_loss": min_loss,
                "best_epoch": best_epoch,
            }
            if ema_decay:
                checkpoint_state["ema_state"] = {
                    k: v.cpu().clone() for k, v in ema_state.items()
                }
            torch.save(
                checkpoint_state,
                str(model_dir / f"checkpoint_epoch_{epoch:03d}.pth"),
            )

        # Validation loss drives model selection whenever validation data exists.
        selection_loss = None
        if val_dataloader is not None and is_checkpoint_epoch:
            net_module.eval()
            val_losses = []
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_gt = val_batch["gt_image"].to(device, non_blocking=True)
                    val_cond = val_batch["cond_image"].to(device, non_blocking=True)
                    # val_mask = val_batch["mask"].to(device)
                    val_mask = None

                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=use_amp,
                    ):
                        val_loss = net_module(val_gt, y_cond=val_cond, mask=val_mask)
                    val_losses.append(val_loss.item())
            avg_val_loss = np.mean(val_losses)
            writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
            print(f"  Validation Loss: {avg_val_loss:.6f}")

            for bucket_name, bucket_loader in val_ng_loaders.items():
                bucket_losses = []
                for val_batch in bucket_loader:
                    val_gt = val_batch["gt_image"].to(device, non_blocking=True)
                    val_cond = val_batch["cond_image"].to(device, non_blocking=True)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=use_amp,
                    ):
                        bucket_loss = net_module(val_gt, y_cond=val_cond, mask=None)
                    bucket_losses.append(bucket_loss.item())
                avg_bucket_loss = np.mean(bucket_losses)
                writer.add_scalar(f"Loss/Validation_{bucket_name}", avg_bucket_loss, epoch)
                print(f"  Validation Loss ({bucket_name}): {avg_bucket_loss:.6f}")

            net_module.train()
            selection_loss = avg_val_loss
        elif val_dataloader is None:
            selection_loss = avg_loss

        if selection_loss is not None and selection_loss < min_loss:
            min_loss = selection_loss
            best_epoch = epoch
            if ema_decay:
                best_model_state = {k: v.cpu().clone() for k, v in ema_state.items()}
            else:
                best_model_state = {
                    k: v.cpu().clone() for k, v in net_module.state_dict().items()
                }
            torch.save(best_model_state, str(model_dir / "best_model.pth"))
            with open(model_dir / "best_epoch.txt", "w") as f:
                f.write(str(best_epoch))
            print(f"  New best model ({selection_loss:.6f})")

        scheduler.step(avg_loss)

        print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.6f}")

    print(f"Training completed. Best model at epoch {best_epoch}")
    writer.close()
    print(f"Results saved to {results_dir}")
    print(f"Models saved to {model_dir}")


def main():

    gpu_ids = [0] if torch.cuda.is_available() else []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config_path = "configs/case1_flumy.json"

    print(f"Config: {config_path}")
    print(f"GPU IDs: {gpu_ids if gpu_ids else 'CPU'}")

    main_worker(config_path, timestamp)


if __name__ == "__main__":
    main()
