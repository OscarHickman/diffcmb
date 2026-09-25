#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska,bluefield1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH -J msgr_inner_conv_L300
#SBATCH -o logs/msgr_inner_conv_L300_%j.out
#SBATCH -e logs/msgr_inner_conv_L300_%j.err
# Not dine2: results/analysis is a symlink into /cosma5, which dine2 does not mount
# (ROADMAP 2026-09-24). Memory sized from past MaxRSS instead of a whole node.

# Phase 0c Step 6 diagnostic: does the messenger sampler's inner Gibbs loop
# converge within n_messenger_iter=100 at production scale? The production
# validation chain (job 11572806-9) completed but did not converge (ROADMAP.md
# Phase 0c Step 6, 2026-07-12) -- this checks the leading hypothesis (inner
# loop under-iterated) before committing to another multi-day production run.

mkdir -p logs

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=""
export TMPDIR="/tmp/diffcmb_msgr_inner_conv_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

echo "=== Messenger inner-loop convergence diagnostic ==="
echo "Python: $($PYTHON --version 2>&1)"
free -h

$PYTHON -u scripts/diagnose_messenger_inner_convergence.py \
    --lmax 300 \
    --nside 256 \
    --data_mode real \
    --data_dir /cosma8/data/dp004/dc-hick2/Plank \
    --checkpoint results/lmax300_nside256_gibbs_real_messenger/checkpoint_chain_1.npz \
    --m_group_size 5 \
    --checkpoints 1 3 10 30 100 300 1000 3000 \
    --out results/analysis/messenger_inner_convergence_L300.npz
