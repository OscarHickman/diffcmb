#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -J debug_phi_grad
#SBATCH -o logs/debug_phi_grad_%j.out
#SBATCH -e logs/debug_phi_grad_%j.err

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
mkdir -p logs

export PYTHONPATH=$(pwd)/diffcmb
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1

echo "=== Debug phi gradient adjoint chain ==="
echo "Node: $(hostname)"
date

.venv/bin/python scripts/debug_phi_grad.py 2>&1
EXIT_CODE=$?
echo "---"
date
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
