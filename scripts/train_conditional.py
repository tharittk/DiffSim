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

import argparse
import json
import os
import socket
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp


def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return str(s.getsockname()[1])
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam, lr_scheduler
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffsim.core.network import Network
from diffsim.models.guided_diffusion import UNet, UNet3D
from diffsim.data import InpaintDatasetCase1, InpaintDatasetCase2, NPYInpaintDataset


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main_worker(rank, world_size, config_path, gpu_ids, port, timestamp):
    """Worker function for each GPU process."""

    # Setup distributed training
    distributed = world_size > 1
    if distributed:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = port
        torch.cuda.set_device(gpu_ids[rank])
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=f'tcp://127.0.0.1:{port}',
            world_size=world_size,
            rank=rank
        )
        device = torch.device(f'cuda:{gpu_ids[rank]}')
        print(f'[Rank {rank}] Using GPU {gpu_ids[rank]} for training')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')

    # Set seed
    set_seed(42 + rank)

    # Only rank 0 prints info
    is_main = (rank == 0)

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

    # Choose module type and predict_type
    module_name = cond.get('module_name', 'guided_diffusion_3d' if is_3d else 'guided_diffusion')
    predict_type = cond.get('predict_type', 'epsilon')

    # Build network
    network = Network(
        unet=unet_config,
        beta_schedule=cond["beta_schedule"],
        module_name=module_name,
        predict_type=predict_type
    )
    network.to(device)

    # Wrap with DDP if distributed
    if distributed:
        network = DDP(network, device_ids=[gpu_ids[rank]], output_device=gpu_ids[rank],
                      find_unused_parameters=True)
        net_module = network.module  # Access underlying module
    else:
        net_module = network

    # Setup noise schedule
    net_module.set_new_noise_schedule(device=device, phase='train')

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
    base_results_dir = Path(output_cfg.get("results_dir", f"./results/{exp_name}_conditional"))
    base_model_dir = Path(output_cfg.get("model_dir", f"./model/{exp_name}_conditional"))
    base_log_dir = Path(output_cfg.get("log_dir", str(base_results_dir / "logs")))

    results_dir = base_results_dir / timestamp
    model_dir = base_model_dir / timestamp
    log_dir = base_log_dir / timestamp

    if is_main:
        results_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))

    if distributed:
        torch.distributed.barrier()  # Wait for rank 0 to create directories

    # Setup dataset
    data_cfg = cond.get("data", config.get("data", {}))
    data_path = data_cfg.get("train_image", data_cfg.get("train_image_path", data_cfg.get("train_path", "")))
    mask_path = data_cfg.get("train_mask", data_cfg.get("mask_path", ""))

    if is_3d:
        dataset = NPYInpaintDataset(
            data_folder=data_path,
            mask_folder=mask_path,
            max_files=12000
        )
    else:
        image_size = config["image_size"]
        DatasetClass = InpaintDatasetCase1 if cond["in_channel"] == 5 else InpaintDatasetCase2
        dataset = DatasetClass(
            data_root=(data_path, mask_path),
            mask_config={'mask_mode': 'file'},
            image_size=[image_size, image_size]
        )

    # Setup sampler and dataloader
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    if is_main:
        print(f"Data path: {data_path}")
        print(f"Mask path: {mask_path}")
        print(f"Dataset class: {dataset.__class__.__name__}")
        print(f"Dataset size: {len(dataset)} samples")
        print(f"Batches per epoch: {len(dataloader)}")
        print(f"World size: {world_size}")

    # Setup optimizer and scheduler
    optimizer = Adam(network.parameters(), lr=train_cfg["lr"])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    # Optional: EMA (only on main process)
    ema_decay = train_cfg.get("ema_decay", None)
    if ema_decay and is_main:
        ema_state = {k: v.clone() for k, v in net_module.state_dict().items()}

    # Training loop
    epochs = train_cfg["epochs"]
    save_every = train_cfg.get("save_every", 500)
    checkpoint_every = train_cfg.get("checkpoint_every", 10)
    min_loss = float('inf')
    best_epoch = 0

    if is_main:
        print(f"Starting conditional training for {exp_name}")
        print(f"Run ID: {timestamp}")
        print(f"Model type: {'3D' if is_3d else '2D'}, Predict type: {predict_type}")
        print(f"Epochs: {epochs}, Batch size: {train_cfg['batch_size']}")
        print(f"Save samples every: {save_every} steps, Checkpoint every: {checkpoint_every} epochs")

    for epoch in range(epochs):
        # Set epoch for sampler (important for shuffling)
        if distributed:
            sampler.set_epoch(epoch)

        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main)

        for step, batch in enumerate(pbar):
            optimizer.zero_grad()

            # Get batch data
            gt_image = batch['gt_image'].to(device)
            cond_image = batch['cond_image']
            if isinstance(cond_image, np.ndarray):
                cond_image = torch.from_numpy(cond_image).float()
            cond_image = cond_image.to(device)
            mask = batch['mask'].to(device)

            # Forward pass
            if distributed:
                loss = network(gt_image, y_cond=cond_image, mask=mask)
            else:
                loss = network(gt_image, y_cond=cond_image, mask=mask)

            epoch_losses.append(loss.item())

            # Backprop
            loss.backward()
            optimizer.step()

            # Update EMA (only on main)
            if ema_decay and is_main:
                with torch.no_grad():
                    for k, v in net_module.state_dict().items():
                        ema_state[k] = ema_decay * ema_state[k] + (1 - ema_decay) * v

            # Logging (only on main)
            global_step = epoch * len(dataloader) + step
            if is_main:
                writer.add_scalar('Loss/Train', loss.item(), global_step)
                writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], global_step)
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

            # Save samples periodically (only on main)
            if is_main and global_step != 0 and global_step % save_every == 0:
                net_module.eval()
                with torch.no_grad():
                    sample_cond = cond_image[:4]
                    sample_gt = gt_image[:4]
                    sample_mask = mask[:4]

                    output, _ = net_module.restoration(sample_cond, y_0=sample_gt, mask=sample_mask)

                    if is_3d:
                        mid = config["image_size"][0] // 2
                        save_image(output[:, :, mid], str(results_dir / f'output_{epoch}_{global_step}.png'), nrow=2, normalize=True, value_range=(-1, 1))
                        save_image(sample_gt[:, :, mid], str(results_dir / f'gt_{epoch}_{global_step}.png'), nrow=2, normalize=True, value_range=(-1, 1))
                    else:
                        save_image(output, str(results_dir / f'output_{epoch}_{global_step}.png'), nrow=2, normalize=True, value_range=(-1, 1))
                        save_image(sample_gt, str(results_dir / f'gt_{epoch}_{global_step}.png'), nrow=2, normalize=True, value_range=(-1, 1))
                net_module.train()

        # Sync losses across processes
        avg_loss = np.mean(epoch_losses)
        if distributed:
            loss_tensor = torch.tensor([avg_loss], device=device)
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / world_size

        # Update best model (only on main)
        if is_main:
            if avg_loss < min_loss:
                min_loss = avg_loss
                best_epoch = epoch
                if ema_decay:
                    best_model_state = {k: v.cpu().clone() for k, v in ema_state.items()}
                else:
                    best_model_state = {k: v.cpu().clone() for k, v in net_module.state_dict().items()}
                torch.save(best_model_state, str(model_dir / 'best_model.pth'))
                with open(model_dir / 'best_epoch.txt', 'w') as f:
                    f.write(str(best_epoch))

            # Save checkpoint
            if epoch % checkpoint_every == 0:
                if ema_decay:
                    save_state = {k: v.cpu().clone() for k, v in ema_state.items()}
                else:
                    save_state = {k: v.cpu().clone() for k, v in net_module.state_dict().items()}
                torch.save(save_state, str(model_dir / f'model_epoch_{epoch:03d}.pth'))

        scheduler.step(avg_loss)

        if is_main:
            print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.6f}")

    if is_main:
        print(f'Training completed. Best model at epoch {best_epoch}')
        writer.close()
        print(f"Results saved to {results_dir}")
        print(f"Models saved to {model_dir}")

    # Cleanup
    if distributed:
        torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train conditional diffusion model")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("-gpu", "--gpu_ids", type=str, default=None, help="GPU ids, e.g., '0,1,2,3'")
    parser.add_argument("-P", "--port", type=str, default="auto", help="Port for distributed training, or 'auto' to find free port")
    args = parser.parse_args()

    # Parse GPU ids
    if args.gpu_ids is not None:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        # After setting CUDA_VISIBLE_DEVICES, GPU indices become 0, 1, 2, ...
        gpu_ids = list(range(len(gpu_ids)))
    else:
        gpu_ids = [0] if torch.cuda.is_available() else []

    # Parse port - auto-select if needed
    if args.port == 'auto':
        port = find_free_port()
        print(f"Auto-selected port: {port}")
    else:
        port = args.port

    world_size = len(gpu_ids) if gpu_ids else 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Config: {args.config}")
    print(f"GPU IDs: {gpu_ids if gpu_ids else 'CPU'}")
    print(f"World size: {world_size}")

    if world_size > 1:
        # Multi-GPU training with DDP
        mp.spawn(
            main_worker,
            nprocs=world_size,
            args=(world_size, args.config, gpu_ids, port, timestamp)
        )
    else:
        # Single GPU or CPU training
        main_worker(0, 1, args.config, gpu_ids, port, timestamp)


if __name__ == "__main__":
    main()
