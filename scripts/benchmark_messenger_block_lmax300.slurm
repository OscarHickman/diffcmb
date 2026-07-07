#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -J msgr_bench_L300
#SBATCH -o logs/msgr_bench_L300_%j.out
#SBATCH -e logs/msgr_bench_L300_%j.err

# ROADMAP.md Phase 0c Step 6: benchmark use_block_correction=True's one-time
# calibration and per-sweep block-Cholesky rebuild cost at lmax=300, real
# Planck data -- the item Step 5 left open before this sampler can replace
# CG in production chains.

mkdir -p logs

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=""
export TMPDIR="/tmp/diffcmb_msgr_bench_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

echo "=== Phase 0c Step 6: messenger block-correction benchmark, lmax=300 ==="
echo "Python: $($PYTHON --version 2>&1)"
free -h

$PYTHON -u scripts/benchmark_messenger_block_lmax300.py
