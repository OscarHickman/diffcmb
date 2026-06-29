#!/bin/bash
#SBATCH -A dp004
#SBATCH -p cosma7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH -J debug_phi_grad2
#SBATCH -o logs/debug_phi_grad2_%j.out
#SBATCH -e logs/debug_phi_grad2_%j.err

cd /cosma/apps/durham/dc-hick2/diffcmb
mkdir -p logs
export PYTHONPATH=$(pwd)/diffcmb
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
echo "=== debug_phi_grad2 ==="
date
.venv/bin/python scripts/debug_phi_grad2.py 2>&1
date
