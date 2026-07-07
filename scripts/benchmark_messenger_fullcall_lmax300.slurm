#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH -J msgr_fullcall_L300
#SBATCH -o logs/msgr_fullcall_L300_%j.out
#SBATCH -e logs/msgr_fullcall_L300_%j.err

# ROADMAP.md Phase 0c Step 6, follow-up to benchmark_messenger_block_lmax300.py:
# times the actual end-to-end sample_alm_messenger call (including the
# n_messenger_iter forward/adjoint SHTs the isolated Cholesky-rebuild
# benchmark did not capture) at lmax=300, real Planck data.
#
# job 11562981 (2026-07-04, --mem=32G) was OOM-killed while calibrating
# m_group_size=5: _calibrate_block_AtA transiently allocates a
# (block_size, npix) dense array per block, ~18GB for the largest
# near-m=0 block at m_group_size=5. Raised to 96G; script now also drops
# each group size's cache after timing it (see benchmark script).

mkdir -p logs

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=""
export TMPDIR="/tmp/diffcmb_msgr_fullcall_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

echo "=== Phase 0c Step 6: messenger full-call benchmark, lmax=300 ==="
echo "Python: $($PYTHON --version 2>&1)"
free -h

$PYTHON -u scripts/benchmark_messenger_fullcall_lmax300.py
