#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -J direct_defl
#SBATCH -o /cosma/apps/durham/dc-hick2/diffcmb/logs/direct_defl_%j.out

cd /cosma/apps/durham/dc-hick2/diffcmb
export PYTHONPATH=$(pwd)/diffcmb
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export TF_ENABLE_ONEDNN_OPTS=0

.venv/bin/python scripts/direct_deflection_test.py 2>&1
