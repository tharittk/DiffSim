# DiffSim: Diffusion Models for Geomodelling

Unconditional and conditional diffusion models for geological facies modeling.

## Overview

DiffSim provides a unified framework for training and using diffusion models on geological data, supporting:
- **2D facies modeling** (channel facies, mud drapes)
- **3D volumetric modeling** (3D facies)
- **Unconditional generation** (generate new samples from noise)
- **Conditional generation / Inpainting** (fill in missing regions given constraints)

## Cases

| Case | Description | Type |
|------|-------------|------|
| case1_geomodeling | 2D channel facies | 2D |
| case2_muddrape | 2D mud drape facies | 2D |
| case3_la3d | 3D facies modeling | 3D |

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
DiffSim/
├── diffsim/                  # Main package
│   ├── models/               # UNet architectures
│   │   ├── unet.py          # 2D UNet
│   │   ├── unet3d.py        # 3D UNet
│   │   └── guided_diffusion/ # Conditional models
│   ├── core/                 # Core utilities
│   │   ├── diffusion.py     # Beta schedules, sampling
│   │   ├── network.py       # Conditional network wrapper
│   │   └── utils.py         # Utility functions
│   └── data/                 # Dataset classes
│       ├── dataset.py       # 2D and 3D datasets
│       └── mask.py          # Mask generation
├── configs/                  # Configuration files
├── notebooks/                # Jupyter notebooks for inference
├── scripts/                  # Training scripts
└── checkpoints/              # Model checkpoints (gitignored)
```

## Checkpoints

Download pretrained models from Google Drive and place them in the `checkpoints/` directory:

| Case | Unconditional | Conditional |
|------|---------------|-------------|
| case1_geomodeling | [Download](#) | [Download](#) |
| case2_muddrape | [Download](#) | [Download](#) |
| case3_la3d | [Download](#) | [Download](#) |

## Usage

### Quick Start

```python
from diffsim import Unet, Diffusion

# Create model
model = Unet(dim=64, channels=1, dim_mults=(1, 2, 4))

# Load checkpoint
model.load_state_dict(torch.load('checkpoints/case1_geomodeling/unconditional.pth'))

# Generate samples
diffusion = Diffusion(timesteps=1500, beta_schedule='linear')
samples = diffusion.sample(model, image_size=64, batch_size=16, channels=1)
```

### Training

**Unconditional Training:**
```bash
python scripts/train_unconditional.py --config configs/case1_geomodeling.json
```

**Conditional Training:**
```bash
python scripts/train_conditional.py --config configs/case1_geomodeling.json
```

### Inference

See the notebooks in `notebooks/` for complete inference examples:
- `case1_geomodeling.ipynb` - 2D channel facies
- `case2_muddrape.ipynb` - 2D mud drape facies
- `case3_la3d.ipynb` - 3D facies modeling

## Configuration

Each case has a JSON configuration file in `configs/` with the following structure:

```json
{
  "name": "case1_geomodeling",
  "type": "2d",
  "image_size": 64,
  "unconditional": {
    "timesteps": 1500,
    "channels": 1,
    "dim": 64,
    "dim_mults": [1, 2, 4],
    "beta_schedule": "linear",
    "training": { ... }
  },
  "conditional": {
    "timesteps": 1500,
    "in_channel": 5,
    "out_channel": 1,
    ...
  }
}
```

## Model Architecture

### Unconditional (Unet / Unet3D)
- Based on the annotated diffusion implementation
- Uses ResNet blocks with attention
- Sinusoidal time embeddings

### Conditional (UNet / UNet3D in guided_diffusion)
- Guided diffusion architecture
- FiLM conditioning with gamma embeddings
- Supports masked training for inpainting

## Key Parameters

| Case | Unconditional | Conditional |
|------|---------------|-------------|
| case1_geomodeling | `dim=64, channels=1, dim_mults=(1,2,4), timesteps=1500` | `in_channel=5, out_channel=1, channel_mults=[1,2,4]` |
| case2_muddrape | `dim=64, channels=1, dim_mults=(1,2,4)` | `in_channel=6, out_channel=1, channel_mults=[1,2,4]` |
| case3_la3d | `dim=64, channels=1, dim_mults=(1,2,4)` | `in_channel=6, out_channel=1, channel_mults=[1,2,4,8]` |

## License

[Your License Here]

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{diffsim2024,
  title={DiffSim: Diffusion Models for Geomodelling},
  author={Your Name},
  year={2024}
}
```
