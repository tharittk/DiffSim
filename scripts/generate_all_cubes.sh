#!/usr/bin/env bash
#
# Generate 3D Flumy facies cubes for all (ng, isbx) combinations.
#
# This iterates over the Cartesian product of ng and isbx ranges
# and calls generate_flumy_cubes.py for each combination.
#
# Usage:
#     bash scripts/generate_all_cubes.sh
#
# Override defaults with environment variables:
#     N_SEEDS=5 OUTPUT_DIR=data/flumy3d/facies bash scripts/generate_all_cubes.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# --- Configurable parameters (override with env vars) ---
NX="${NX:-256}"
NY="${NY:-256}"
MESH="${MESH:-20}"
HMAX="${HMAX:-10.0}"
DZ="${DZ:-1.0}"
N_SEEDS="${N_SEEDS:-5}"
SEED_START="${SEED_START:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-data/flumy3d/facies}"

# Net-to-gross range
NG_VALUES="${NG_VALUES:-10 20 30 40 50 60 70 80 90}"
# Sand body extension range
ISBX_VALUES="${ISBX_VALUES:-20 30 40 50 60 70 80 90 100 110 120 130 140 150 160}"

# ---

echo "=== Flumy 3D facies cube generation ==="
echo "Grid: ${NX}x${NY}, mesh=${MESH}m, hmax=${HMAX}m, dz=${DZ}m"
echo "Seeds per combo: ${N_SEEDS} (starting at ${SEED_START})"
echo "Output: ${OUTPUT_DIR}"
echo "ng values:   ${NG_VALUES}"
echo "isbx values: ${ISBX_VALUES}"
echo ""

cd "$PROJECT_ROOT"

total=0
for ng in $NG_VALUES; do
    for isbx in $ISBX_VALUES; do
        total=$((total + 1))
    done
done

count=0
for ng in $NG_VALUES; do
    for isbx in $ISBX_VALUES; do
        count=$((count + 1))
        echo "[${count}/${total}] ng=${ng}, isbx=${isbx}"
        python scripts/generate_flumy_cubes.py \
            --nx "$NX" --ny "$NY" --mesh "$MESH" --hmax "$HMAX" --dz "$DZ" \
            --ng "$ng" --isbx "$isbx" \
            --n_seeds "$N_SEEDS" --seed_start "$SEED_START" \
            --output_dir "$OUTPUT_DIR"
        echo ""
    done
done

echo "=== Done. Total combinations: ${total} ==="
