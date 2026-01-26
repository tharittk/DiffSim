#!/bin/bash
# Train all diffusion models for 3 cases
# Usage: bash scripts/train_all.sh

set -e  # Exit on error

echo "=========================================="
echo "DiffSim Training Script"
echo "=========================================="

# Case 1: 2D Channel Facies
echo ""
echo "[Case 1] 2D Channel Facies - Unconditional"
python scripts/train_unconditional.py --config configs/case1_geomodeling.json

echo ""
echo "[Case 1] 2D Channel Facies - Conditional"
python scripts/train_conditional.py --config configs/case1_geomodeling.json

# Case 2: 2D Mud Drape
echo ""
echo "[Case 2] 2D Mud Drape - Unconditional"
python scripts/train_unconditional.py --config configs/case2_muddrape.json

echo ""
echo "[Case 2] 2D Mud Drape - Conditional"
python scripts/train_conditional.py --config configs/case2_muddrape.json

# Case 3: 3D LA
echo ""
echo "[Case 3] 3D LA - Unconditional"
python scripts/train_unconditional.py --config configs/case3_la3d.json

echo ""
echo "[Case 3] 3D LA - Conditional"
python scripts/train_conditional.py --config configs/case3_la3d.json

echo ""
echo "=========================================="
echo "All training completed!"
echo "=========================================="
