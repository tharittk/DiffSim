#!/usr/bin/env python
"""
Verification script to confirm the installation is correct.

Run this script to verify:
1. All imports work correctly
2. Models can be instantiated
3. Forward passes work (inference)
4. Training steps work (backward passes)

Usage:
    python tests/test_models.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

def print_status(name, success, error=None):
    if success:
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name}")
        if error:
            print(f"         Error: {error}")

def test_imports():
    """Test all package imports."""
    print("\n=== Testing Imports ===")

    tests = [
        ("diffsim main package", "from diffsim import Unet, Unet3D, Network"),
        ("diffusion module", "from diffsim.core.diffusion import Diffusion, sample, linear_beta_schedule"),
        ("guided_diffusion 2D", "from diffsim.models.guided_diffusion import UNet"),
        ("guided_diffusion 3D", "from diffsim.models.guided_diffusion import UNet3D"),
        ("data module", "from diffsim.data import ImageDataset, NPYDataset, bbox2mask"),
        ("core utilities", "from diffsim.core.utils import tensor2img, set_seed"),
        ("network module", "from diffsim.core.network import Network, make_beta_schedule"),
    ]

    all_passed = True
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print_status(name, True)
        except Exception as e:
            print_status(name, False, str(e))
            all_passed = False

    return all_passed

def test_unconditional_2d():
    """Test 2D unconditional model."""
    print("\n=== Testing 2D Unconditional Model ===")

    from diffsim.models import Unet
    from diffsim.core.diffusion import Diffusion

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_passed = True

    # Test model instantiation
    try:
        model = Unet(dim=64, channels=1, dim_mults=(1, 2, 4))
        model.to(device)
        print_status("Model instantiation", True)
    except Exception as e:
        print_status("Model instantiation", False, str(e))
        return False

    # Test forward pass (inference)
    try:
        x = torch.randn(2, 1, 64, 64, device=device)
        t = torch.randint(0, 1000, (2,), device=device)
        with torch.no_grad():
            out = model(x, t)
        assert out.shape == x.shape, f"Output shape mismatch: {out.shape} vs {x.shape}"
        print_status("Forward pass (inference)", True)
    except Exception as e:
        print_status("Forward pass (inference)", False, str(e))
        all_passed = False

    # Test backward pass (training)
    try:
        diffusion = Diffusion(timesteps=1000, beta_schedule='linear')
        x = torch.randn(2, 1, 64, 64, device=device)
        t = torch.randint(0, 1000, (2,), device=device)
        loss = diffusion.p_losses(model, x, t, loss_type='huber')
        loss.backward()
        print_status("Backward pass (training)", True)
    except Exception as e:
        print_status("Backward pass (training)", False, str(e))
        all_passed = False

    # Test sampling
    try:
        model.eval()
        with torch.no_grad():
            # Quick sample with few steps for testing
            diffusion_small = Diffusion(timesteps=10, beta_schedule='linear')
            samples = diffusion_small.sample(model, image_size=64, batch_size=1, channels=1)
        assert samples.shape == (1, 1, 64, 64), f"Sample shape mismatch: {samples.shape}"
        print_status("Sampling", True)
    except Exception as e:
        print_status("Sampling", False, str(e))
        all_passed = False

    return all_passed

def test_unconditional_3d():
    """Test 3D unconditional model."""
    print("\n=== Testing 3D Unconditional Model ===")

    from diffsim.models import Unet3D
    from diffsim.core.diffusion import Diffusion

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_passed = True

    # Test model instantiation
    try:
        model = Unet3D(dim=32, channels=1, dim_mults=(1, 2))
        model.to(device)
        print_status("Model instantiation", True)
    except Exception as e:
        print_status("Model instantiation", False, str(e))
        return False

    # Test forward pass (inference)
    try:
        x = torch.randn(1, 1, 16, 16, 16, device=device)
        t = torch.randint(0, 100, (1,), device=device)
        with torch.no_grad():
            out = model(x, t)
        assert out.shape == x.shape, f"Output shape mismatch: {out.shape} vs {x.shape}"
        print_status("Forward pass (inference)", True)
    except Exception as e:
        print_status("Forward pass (inference)", False, str(e))
        all_passed = False

    # Test backward pass (training)
    try:
        diffusion = Diffusion(timesteps=100, beta_schedule='linear')
        x = torch.randn(1, 1, 16, 16, 16, device=device)
        t = torch.randint(0, 100, (1,), device=device)
        loss = diffusion.p_losses(model, x, t, loss_type='huber')
        loss.backward()
        print_status("Backward pass (training)", True)
    except Exception as e:
        print_status("Backward pass (training)", False, str(e))
        all_passed = False

    return all_passed

def test_conditional_2d():
    """Test 2D conditional model."""
    print("\n=== Testing 2D Conditional Model ===")

    from diffsim.core.network import Network

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_passed = True

    unet_config = {
        "image_size": 64,
        "in_channel": 5,
        "out_channel": 1,
        "inner_channel": 32,
        "channel_mults": [1, 2],
        "attn_res": [16],
        "res_blocks": 1,
        "dropout": 0.0,
    }

    beta_schedule = {
        "train": {"schedule": "linear", "n_timestep": 100, "linear_start": 1e-6, "linear_end": 0.01},
        "test": {"schedule": "linear", "n_timestep": 100, "linear_start": 1e-6, "linear_end": 0.01}
    }

    # Test model instantiation
    try:
        network = Network(unet=unet_config, beta_schedule=beta_schedule, module_name='guided_diffusion')
        network.to(device)
        network.set_new_noise_schedule(device=torch.device(device), phase='train')
        network.set_loss(nn.MSELoss())
        print_status("Network instantiation", True)
    except Exception as e:
        print_status("Network instantiation", False, str(e))
        return False

    # Test forward pass (training)
    try:
        y_0 = torch.randn(2, 1, 64, 64, device=device)
        y_cond = torch.randn(2, 4, 64, 64, device=device)
        mask = torch.ones(2, 1, 64, 64, device=device)
        loss = network(y_0, y_cond=y_cond, mask=mask)
        loss.backward()
        print_status("Forward + backward pass (training)", True)
    except Exception as e:
        print_status("Forward + backward pass (training)", False, str(e))
        all_passed = False

    # Test restoration (inference)
    try:
        network.eval()
        network.set_new_noise_schedule(device=torch.device(device), phase='test')
        y_cond = torch.randn(1, 4, 64, 64, device=device)
        y_0 = torch.randn(1, 1, 64, 64, device=device)
        mask = torch.ones(1, 1, 64, 64, device=device)

        # Use fewer timesteps for testing
        network.num_timesteps = 10
        with torch.no_grad():
            output, _ = network.restoration(y_cond, y_0=y_0, mask=mask, sample_num=2)
        assert output.shape == (1, 1, 64, 64), f"Output shape mismatch: {output.shape}"
        print_status("Restoration (inference)", True)
    except Exception as e:
        print_status("Restoration (inference)", False, str(e))
        all_passed = False

    return all_passed

def test_conditional_3d():
    """Test 3D conditional model."""
    print("\n=== Testing 3D Conditional Model ===")

    from diffsim.core.network import Network

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_passed = True

    unet_config = {
        "image_size": 16,
        "in_channel": 3,
        "out_channel": 1,
        "inner_channel": 16,
        "channel_mults": [1, 2],
        "attn_res": [8],
        "res_blocks": 1,
        "dropout": 0.0,
    }

    beta_schedule = {
        "train": {"schedule": "linear", "n_timestep": 50, "linear_start": 1e-6, "linear_end": 0.01},
        "test": {"schedule": "linear", "n_timestep": 50, "linear_start": 1e-6, "linear_end": 0.01}
    }

    # Test model instantiation
    try:
        network = Network(unet=unet_config, beta_schedule=beta_schedule, module_name='guided_diffusion_3d')
        network.to(device)
        network.set_new_noise_schedule(device=torch.device(device), phase='train')
        network.set_loss(nn.MSELoss())
        print_status("Network instantiation", True)
    except Exception as e:
        print_status("Network instantiation", False, str(e))
        return False

    # Test forward pass (training)
    try:
        y_0 = torch.randn(1, 1, 16, 16, 16, device=device)
        y_cond = torch.randn(1, 2, 16, 16, 16, device=device)
        mask = torch.ones(1, 1, 16, 16, 16, device=device)
        loss = network(y_0, y_cond=y_cond, mask=mask)
        loss.backward()
        print_status("Forward + backward pass (training)", True)
    except Exception as e:
        print_status("Forward + backward pass (training)", False, str(e))
        all_passed = False

    return all_passed

def test_data_utilities():
    """Test data utilities."""
    print("\n=== Testing Data Utilities ===")

    from diffsim.data import bbox2mask, brush_stroke_mask, random_bbox
    import numpy as np

    all_passed = True

    # Test bbox2mask
    try:
        mask = bbox2mask((64, 64), (10, 10, 20, 20))
        assert mask.shape == (64, 64, 1), f"Mask shape mismatch: {mask.shape}"
        assert mask.sum() == 20 * 20, f"Mask area mismatch"
        print_status("bbox2mask", True)
    except Exception as e:
        print_status("bbox2mask", False, str(e))
        all_passed = False

    # Test brush_stroke_mask
    try:
        mask = brush_stroke_mask((64, 64))
        assert mask.shape == (64, 64, 1), f"Mask shape mismatch: {mask.shape}"
        print_status("brush_stroke_mask", True)
    except Exception as e:
        print_status("brush_stroke_mask", False, str(e))
        all_passed = False

    return all_passed

def main():
    print("=" * 60)
    print("DiffSim Installation Verification")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: {device=}")
    results = {}

    # Run all tests
    results['imports'] = test_imports()
    results['unconditional_2d'] = test_unconditional_2d()
    results['unconditional_3d'] = test_unconditional_3d()
    results['conditional_2d'] = test_conditional_2d()
    results['conditional_3d'] = test_conditional_3d()
    results['data_utilities'] = test_data_utilities()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED - Installation verified!")
    else:
        print("SOME TESTS FAILED - Please check the errors above.")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
