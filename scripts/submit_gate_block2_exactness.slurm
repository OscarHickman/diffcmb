#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J gate2_block2_exactness
#SBATCH -o logs/gate2_block2_exactness_%j.out
#SBATCH -e logs/gate2_block2_exactness_%j.err
#SBATCH --exclusive

# Phase 2 pre-production gate 2 (ROADMAP.md Section 1): Block-2 exactness
# experiment at lmax=50, nside=64 -- does an exact Gaussian Block-2 (alm)
# draw calibrated against the unlensed operator survive weak lensing, or
# does it bias C_l^TT once phi is genuinely in the data?
LMAX=50
NSIDE=64
BURNIN=500
SAMPLES=3000

# results/analysis is a symlink into /cosma5, which dine2 does not mount (ROADMAP
# 2026-09-24). Fail loudly here rather than at the first write.
if [ ! -d /cosma/apps/durham/dc-hick2/diffcmb/results/analysis/ ]; then
    echo "results/analysis (/cosma5) is not mounted on $(hostname); relocate outputs before running on dine2" >&2
    exit 1
fi
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

mkdir -p logs results/analysis

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "Starting Phase 2 gate 2: Block-2 exactness check (lmax=${LMAX}, nside=${NSIDE})"
echo "Using Python: $($PYTHON --version 2>&1)"
free -h

$PYTHON -u scripts/gate_block2_exactness_check.py \
    --lmax $LMAX \
    --nside $NSIDE \
    --n_burnin $BURNIN \
    --n_samples $SAMPLES \
    --out results/analysis/block2_exactness_check_lmax${LMAX}.npz
