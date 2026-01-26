#!/usr/bin/bash
#SBATCH --job-name=diffsim_test
#SBATCH --output=diffsim_test_%j.out
#SBATCH --error=diffsim_test_%j.err
#SBATCH -c 10
#SBATCH -p serc --gres gpu:2
#SBATCH -C GPU_MEM:80GB
#SBATCH --time=1:00:00

source /home/users/${USER}/.bashrc
source activate /scratch/users/minghuix/conda_envs/ddpm2
ml reset
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# Create test config (5 epochs)
python3 -c "
import json
with open('configs/case3_la3d.json') as f: cfg = json.load(f)
cfg['conditional']['training']['epochs'] = 5
cfg['conditional']['training']['save_every'] = 10
with open('configs/test_multigpu.json', 'w') as f: json.dump(cfg, f, indent=2)
"

# Run with 4 GPUs
python3 scripts/train_conditional.py -c configs/test_multigpu.json -gpu 0,1 -P auto
