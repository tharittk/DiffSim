#!/usr/bin/env python
"""
Unified unconditional diffusion training script.

Usage:
    python train_unconditional.py --config configs/case1_geomodeling.json

Supports both 2D and 3D cases based on config["type"].
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffsim.models import Unet, Unet3D
from diffsim.core.diffusion import Diffusion
from diffsim.data import ImageDataset, NPYDataset
from diffsim.core.utils import num_to_groups


def main(config_path):
    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Determine 2D or 3D
    is_3d = config["type"] == "3d"
    Model = Unet3D if is_3d else Unet

    # Extract unconditional config
    uncond = config["unconditional"]
    train_cfg = uncond["training"]

    # Build model
    model = Model(
        dim=uncond["dim"],
        channels=uncond["channels"],
        dim_mults=tuple(uncond["dim_mults"])
    )

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Initialize diffusion
    diffusion = Diffusion(
        timesteps=uncond["timesteps"],
        beta_schedule=uncond["beta_schedule"]
    )

    # Setup output directories
    exp_name = config["name"]
    results_dir = Path(f"./results/{exp_name}")
    model_dir = Path(f"./model/{exp_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Setup tensorboard
    writer = SummaryWriter(log_dir=str(results_dir / "logs"))

    # Setup dataset
    data_path = config["data"]["train_path"]
    if is_3d:
        image_size = config["image_size"]  # List [D, H, W]
        dataset = NPYDataset(folder=data_path, max_files=12000)
    else:
        image_size = config["image_size"]  # int
        dataset = ImageDataset(
            data_root=data_path,
            image_size=[image_size, image_size],
            convert_image_to='L' if uncond["channels"] == 1 else 'RGB'
        )

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Setup optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=train_cfg["lr"])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    # Training loop
    epochs = train_cfg["epochs"]
    min_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    save_every = 500

    print(f"Starting training for {exp_name}")
    print(f"Model type: {'3D' if is_3d else '2D'}")
    print(f"Epochs: {epochs}, Batch size: {train_cfg['batch_size']}")
    print(f"Device: {device}")

    for epoch in range(epochs):
        epoch_losses = []
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch = batch.to(device)
            batch_size = batch.shape[0]

            # Sample random timesteps
            t = torch.randint(0, uncond["timesteps"], (batch_size,), device=device).long()

            # Compute loss
            loss = diffusion.p_losses(model, batch, t, loss_type=train_cfg["loss"])
            epoch_losses.append(loss.item())

            # Backprop
            loss.backward()
            optimizer.step()

            # Logging
            global_step = epoch * len(dataloader) + step
            writer.add_scalar('Loss/Train', loss.item(), global_step)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], global_step)

            if global_step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}: Loss = {loss.item():.6f}")

            # Save samples periodically
            if global_step != 0 and global_step % save_every == 0:
                model.eval()
                with torch.no_grad():
                    if is_3d:
                        samples = diffusion.sample(model, image_size, batch_size=4, channels=uncond["channels"])
                        # Save middle slice
                        mid_slice = samples[:, :, image_size[0] // 2]
                        save_image(mid_slice, str(results_dir / f'sample_{epoch}_{global_step}.png'), nrow=2)
                    else:
                        samples = diffusion.sample(model, image_size, batch_size=16, channels=uncond["channels"])
                        save_image(samples, str(results_dir / f'sample_{epoch}_{global_step}.png'), nrow=4)
                model.train()

        # Update best model
        avg_loss = np.mean(epoch_losses)
        if avg_loss < min_loss:
            min_loss = avg_loss
            best_model_state = model.state_dict()
            best_epoch = epoch
            torch.save(best_model_state, str(model_dir / 'best_model.pth'))
            with open(model_dir / 'best_epoch.txt', 'w') as f:
                f.write(str(best_epoch))

        # Save checkpoint
        if epoch % 10 == 0:
            torch.save(model.state_dict(), str(model_dir / f'model_epoch_{epoch:03d}.pth'))

        scheduler.step(avg_loss)
        print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.6f}")

    print(f'Training completed. Best model at epoch {best_epoch}')

    # Final sample generation
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        if is_3d:
            final_samples = diffusion.sample(model, image_size, batch_size=8, channels=uncond["channels"])
        else:
            final_samples = diffusion.sample(model, image_size, batch_size=64, channels=uncond["channels"])
        save_image(
            final_samples if not is_3d else final_samples[:, :, image_size[0] // 2],
            str(results_dir / 'final_samples.png'),
            nrow=8
        )

    writer.close()
    print(f"Results saved to {results_dir}")
    print(f"Models saved to {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train unconditional diffusion model")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    args = parser.parse_args()
    main(args.config)
