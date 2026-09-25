#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=6:00:00
#SBATCH -J smoke_lmax300_mf_lensing
#SBATCH -o logs/smoke_lmax300_mf_lensing_%j.out
#SBATCH -e logs/smoke_lmax300_mf_lensing_%j.err
#SBATCH --exclusive

# ROADMAP.md Section 1, "Next up" item 1: lmax=300 smoke run of the
# matrix-free Block 3 lensing port (commit 19a7977). Short chain, warm-started
# at truth -- confirms the port holds at production scale (finite
# samples/logp, sane accept rate, tractable per-sweep cost) before committing
# to the full simulation-validation chain (item 2).
LMAX=300
NSIDE=256
BURNIN=20
SAMPLES=50

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

echo "Starting lmax=300 matrix-free Block 3 lensing smoke run (nside=${NSIDE})"
echo "Using Python: $($PYTHON --version 2>&1)"
free -h

$PYTHON -u scripts/smoke_lmax300_matrixfree_lensing.py \
    --lmax $LMAX \
    --nside $NSIDE \
    --n_burnin $BURNIN \
    --n_samples $SAMPLES \
    --out results/analysis/smoke_lmax300_matrixfree_lensing.npz
