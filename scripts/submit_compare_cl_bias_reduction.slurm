#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska,bluefield1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -J compare_cl_bias
#SBATCH -o logs/compare_cl_bias_%j.out
#SBATCH -e logs/compare_cl_bias_%j.err
# Not dine2: results/analysis is a symlink into /cosma5, which dine2 does not mount
# (ROADMAP 2026-09-24). Memory sized from past MaxRSS instead of a whole node.

# 2026-09-11. The C_l^TT bias-reduction comparison (ROADMAP.md section 2).
# Analysis only; loads ~900 MB of chains, so it runs under SLURM rather than
# on the login node. Both inputs are packing-v2: the lensing-aware chain is
# job 11966631 and the lensing-blind baseline is job 11980570 (the 2026-08-12
# baseline is old-2L-packed and would be a different model -- the script
# checks widths and refuses on a mismatch).

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
mkdir -p logs results/analysis
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0

.venv/bin/python -u scripts/compare_cl_bias_reduction.py \
    --blind results/analysis/lensing_blind_baseline_lmax128_packingv2.npz \
    --aware results/analysis/pilot_coverage_lmax128_postfix_hmc.npz \
    --lmax 128 \
    --out results/analysis/figures/cl_bias_reduction_lmax128

rc=$?
echo "Exited with status $rc"
# Propagate the status so sacct shows FAILED rather than COMPLETED.
exit $rc
