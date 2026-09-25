#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J gate_patch_mixing
#SBATCH -o logs/gate_patch_mixing_%j.out
#SBATCH -e logs/gate_patch_mixing_%j.err
#SBATCH --exclusive

# ROADMAP.md Section 1, CMBLensing.jl benchmark mandatory first step: a
# patch-scale (f_sky~0.016) mixing sanity check of the 3-block Gibbs sampler
# at production lmax=300, before attempting the matched flat-patch benchmark.
# Warm-started at truth so the diagnostic isolates mixing from burn-in-to-mode
# time (mirrors gate_phi_alm_mixing_check.py's structure).
LMAX=300
NSIDE=256
FSKY=0.016
BURNIN=50
SAMPLES=150

# results/analysis is a symlink into /cosma5, which dine2 does not mount (ROADMAP
# 2026-09-24). Fail loudly here rather than at the first write.
if [ ! -d /cosma/apps/durham/dc-hick2/diffcmb/results/analysis/ ]; then
    echo "results/analysis (/cosma5) is not mounted on $(hostname); relocate outputs before running on dine2" >&2
    exit 1
fi
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

mkdir -p logs results/analysis
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "Starting patch-scale (f_sky=${FSKY}) mixing check (lmax=${LMAX}, nside=${NSIDE})"
echo "Using Python: $($PYTHON --version 2>&1)"
free -h

$PYTHON -u scripts/gate_patch_mixing_check.py \
    --lmax $LMAX \
    --nside $NSIDE \
    --f_sky $FSKY \
    --n_burnin $BURNIN \
    --n_samples $SAMPLES \
    --checkpoint_path results/analysis/gate_patch_mixing_ckpt.npz \
    --out results/analysis/gate_patch_mixing_check.npz
