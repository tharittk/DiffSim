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


from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: E402
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


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    # Set seed
    set_seed(42)

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Extract conditional config
    cond = config["conditional"]
    train_cfg = cond["training"]

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
            num_workers=4,
            pin_memory=True,
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
            num_workers=4,
            pin_memory=True,
        )

    print(f"Data path: {data_path}")
    print(f"Dataset class: {dataset.__class__.__name__}")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batches per epoch: {len(dataloader)}")

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
            num_workers=2,
            pin_memory=True,
        )

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
            optimizer.zero_grad()

            # Get batch data
            gt_image = batch["gt_image"].to(device)
            cond_image = batch["cond_image"]
            if isinstance(cond_image, np.ndarray):
                cond_image = torch.from_numpy(cond_image).float()
            cond_image = cond_image.to(device)

            # there is no well locations
            # mask = batch["mask"].to(device)
            mask = None

            loss = network(gt_image, y_cond=cond_image, mask=mask)

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

        # Update best model (only on main)
        if avg_loss < min_loss:
            min_loss = avg_loss
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

        # Save checkpoint (with full state for resumability)
        if epoch % checkpoint_every == 0:
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

        # Validation loss (only on main process)
        if val_dataloader is not None and epoch % checkpoint_every == 0:
            net_module.eval()
            val_losses = []
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_gt = val_batch["gt_image"].to(device)
                    val_cond = val_batch["cond_image"].to(device)
                    # val_mask = val_batch["mask"].to(device)
                    val_mask = None

                    val_loss = net_module(val_gt, y_cond=val_cond, mask=val_mask)
                    val_losses.append(val_loss.item())
            avg_val_loss = np.mean(val_losses)
            writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
            print(f"  Validation Loss: {avg_val_loss:.6f}")
            net_module.train()

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
