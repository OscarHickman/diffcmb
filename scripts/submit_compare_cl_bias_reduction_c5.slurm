#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -J compare_cl_bias
#SBATCH -o logs/compare_cl_bias_%j.out
#SBATCH -e logs/compare_cl_bias_%j.err

# 2026-09-11. The C_l^TT bias-reduction comparison (ROADMAP.md section 2).
# Analysis only; loads ~900 MB of chains, so it runs under SLURM rather than
# on the login node. Runs on cosma5 (not dine2) because the Block-4-ON
# ensemble extension saturates dine2; both inputs are real files on NFS, so
# no /cosma5 mount is required either way.
# on the login node. Both inputs are packing-v2: the lensing-aware chain is
# job 11966631 and the lensing-blind baseline is job 11980570 (the 2026-08-12
# baseline is old-2L-packed and would be a different model -- the script
# checks widths and refuses on a mismatch).

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
mkdir -p logs results/analysis
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=4
export TF_ENABLE_ONEDNN_OPTS=0

.venv/bin/python -u scripts/compare_cl_bias_reduction.py \
    --blind results/analysis/lensing_blind_baseline_lmax128_packingv2.npz \
    --aware results/analysis/pilot_coverage_lmax128_postfix_hmc.npz \
    --lmax 128 \
    --exclude_unreliable \
    --out results/analysis/figures/cl_bias_reduction_lmax128_paper

echo "Exited with status $?"
