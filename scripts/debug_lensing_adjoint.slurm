#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH -J debug_lensing_adj
#SBATCH -o logs/debug_lensing_adj_%j.out
#SBATCH -e logs/debug_lensing_adj_%j.err

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
mkdir -p logs

export PYTHONPATH=$(pwd)/diffcmb
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1

echo "=== Debug lensing adjoint (pure numpy, no TF) ==="
echo "Node: $(hostname)"
date

.venv/bin/python scripts/debug_lensing_adjoint.py 2>&1
EXIT_CODE=$?
echo "---"
date
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
