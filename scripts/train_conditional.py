#!/usr/bin/env python
"""
Unified conditional diffusion training script.

Usage:
    python train_conditional.py --config configs/case1_geomodeling.json

Supports both 2D and 3D cases based on config["type"].
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffsim.core.network import Network
from diffsim.models.guided_diffusion import UNet, UNet3D
from diffsim.data import InpaintDataset, NPYInpaintDataset


def main(config_path):
    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Determine 2D or 3D
    is_3d = config["type"] == "3d"

    # Extract conditional config
    cond = config["conditional"]
    train_cfg = cond["training"]

    # Build UNet config
    unet_config = {
        "image_size": config["image_size"] if not is_3d else config["image_size"][0],
        "in_channel": cond["in_channel"],
        "out_channel": cond["out_channel"],
        "inner_channel": cond["inner_channel"],
        "channel_mults": cond["channel_mults"],
        "attn_res": cond["attn_res"],
        "res_blocks": cond["res_blocks"],
        "dropout": cond["dropout"],
    }

    # Choose module type
    module_name = 'guided_diffusion_3d' if is_3d else 'guided_diffusion'

    # Build network
    network = Network(
        unet=unet_config,
        beta_schedule=cond["beta_schedule"],
        module_name=module_name
    )

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network.to(device)

    # Setup noise schedule
    network.set_new_noise_schedule(device=torch.device(device), phase='train')

    # Set loss function
    if train_cfg["loss"] == "mse":
        loss_fn = nn.MSELoss()
    elif train_cfg["loss"] == "l1":
        loss_fn = nn.L1Loss()
    else:
        loss_fn = nn.MSELoss()
    network.set_loss(loss_fn)

    # Setup output directories
    exp_name = config["name"]
    results_dir = Path(f"./results/{exp_name}_conditional")
    model_dir = Path(f"./model/{exp_name}_conditional")
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Setup tensorboard
    writer = SummaryWriter(log_dir=str(results_dir / "logs"))

    # Setup dataset
    data_path = config["data"]["train_path"]
    mask_path = config["data"]["mask_path"]

    if is_3d:
        dataset = NPYInpaintDataset(
            data_folder=data_path,
            mask_folder=mask_path,
            max_files=12000
        )
    else:
        image_size = config["image_size"]
        dataset = InpaintDataset(
            data_root=(data_path, mask_path),
            mask_config={'mask_mode': 'file'},
            image_size=[image_size, image_size]
        )

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Setup optimizer and scheduler
    optimizer = Adam(network.parameters(), lr=train_cfg["lr"])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    # Optional: EMA
    ema_decay = train_cfg.get("ema_decay", None)
    if ema_decay:
        ema_state = {k: v.clone() for k, v in network.state_dict().items()}

    # Training loop
    epochs = train_cfg["epochs"]
    min_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    save_every = 500

    print(f"Starting conditional training for {exp_name}")
    print(f"Model type: {'3D' if is_3d else '2D'}")
    print(f"Epochs: {epochs}, Batch size: {train_cfg['batch_size']}")
    print(f"Device: {device}")

    for epoch in range(epochs):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(pbar):
            optimizer.zero_grad()

            # Get batch data
            gt_image = batch['gt_image'].to(device)
            cond_image = batch['cond_image'].to(device)
            mask = batch['mask'].to(device)

            # Handle tensor type for cond_image
            if isinstance(cond_image, np.ndarray):
                cond_image = torch.from_numpy(cond_image).float()

            # Forward pass
            loss = network(gt_image, y_cond=cond_image, mask=mask)
            epoch_losses.append(loss.item())

            # Backprop
            loss.backward()
            optimizer.step()

            # Update EMA
            if ema_decay:
                with torch.no_grad():
                    for k, v in network.state_dict().items():
                        ema_state[k] = ema_decay * ema_state[k] + (1 - ema_decay) * v

            # Logging
            global_step = epoch * len(dataloader) + step
            writer.add_scalar('Loss/Train', loss.item(), global_step)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], global_step)

            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

            # Save samples periodically
            if global_step != 0 and global_step % save_every == 0:
                network.eval()
                with torch.no_grad():
                    # Take a subset for visualization
                    sample_cond = cond_image[:4]
                    sample_gt = gt_image[:4]
                    sample_mask = mask[:4]

                    # Generate samples
                    output, _ = network.restoration(sample_cond, y_0=sample_gt, mask=sample_mask)

                    if is_3d:
                        # Save middle slice
                        mid = config["image_size"][0] // 2
                        save_image(output[:, :, mid], str(results_dir / f'output_{epoch}_{global_step}.png'), nrow=2)
                        save_image(sample_gt[:, :, mid], str(results_dir / f'gt_{epoch}_{global_step}.png'), nrow=2)
                    else:
                        save_image(output, str(results_dir / f'output_{epoch}_{global_step}.png'), nrow=2)
                        save_image(sample_gt, str(results_dir / f'gt_{epoch}_{global_step}.png'), nrow=2)
                network.train()

        # Update best model
        avg_loss = np.mean(epoch_losses)
        if avg_loss < min_loss:
            min_loss = avg_loss
            best_epoch = epoch
            if ema_decay:
                best_model_state = ema_state.copy()
            else:
                best_model_state = {k: v.clone() for k, v in network.state_dict().items()}
            torch.save(best_model_state, str(model_dir / 'best_model.pth'))
            with open(model_dir / 'best_epoch.txt', 'w') as f:
                f.write(str(best_epoch))

        # Save checkpoint
        if epoch % 10 == 0:
            save_state = ema_state if ema_decay else network.state_dict()
            torch.save(save_state, str(model_dir / f'model_epoch_{epoch:03d}.pth'))

        scheduler.step(avg_loss)
        print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.6f}")

    print(f'Training completed. Best model at epoch {best_epoch}')
    writer.close()
    print(f"Results saved to {results_dir}")
    print(f"Models saved to {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train conditional diffusion model")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    args = parser.parse_args()
    main(args.config)
