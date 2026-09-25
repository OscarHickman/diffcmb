#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J validate_sim_lmax300
#SBATCH -o logs/validate_sim_lmax300_%j.out
#SBATCH -e logs/validate_sim_lmax300_%j.err
#SBATCH --exclusive

# ROADMAP.md Section 1, "Next up" item 1: full lmax=300 simulation validation
# of the matrix-free 3-block Gibbs sampler, now that the Block 3 lensing port
# has passed its production-scale smoke test (job 11611338). n_lfs=10 for
# both HMC blocks per the tuning check (job 11612586, ~29s/sweep). Checkpoints
# every 50 sweeps to results/analysis/validate_sim_lmax300_ckpt.npz so the run
# can resume across multiple 24h submissions if it doesn't finish in one.
LMAX=300
NSIDE=256
BURNIN=200
SAMPLES=1000
N_LFS=10
PHI_N_LFS=10

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

echo "Starting lmax=300 simulation validation run (nside=${NSIDE})"
echo "Using Python: $($PYTHON --version 2>&1)"
free -h

$PYTHON -u scripts/validate_sim_lmax300_lensing.py \
    --lmax $LMAX \
    --nside $NSIDE \
    --n_burnin $BURNIN \
    --n_samples $SAMPLES \
    --n_lfs $N_LFS \
    --phi_n_lfs $PHI_N_LFS \
    --checkpoint_path results/analysis/validate_sim_lmax300_ckpt.npz \
    --checkpoint_every 50 \
    --out results/analysis/validate_sim_lmax300.npz
