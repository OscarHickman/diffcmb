#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH -J lens_power_transfer
#SBATCH -o logs/lens_power_transfer_%j.out
#SBATCH -e logs/lens_power_transfer_%j.err
#SBATCH --exclusive

# 2026-09-22. Decomposes figure 2's lensing-blind C_l^TT deficit into
# physical lensing, band-limit truncation, and the bilinear-interpolation
# operator (scripts/validate_lensing_power_transfer.py). Must be read before
# figure 2 is captioned or its deficit is described as "lensing bias".

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
mkdir -p logs results/analysis
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=32

.venv/bin/python -u scripts/validate_lensing_power_transfer.py \
    --lmax_list 64 128 192 --n_real 16 --nthreads 32 \
    --out results/analysis/lensing_power_transfer.npz

echo "Exited with status $?"
