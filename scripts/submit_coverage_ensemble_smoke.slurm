#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska,bluefield1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -J cov_ens_smoke
#SBATCH -o logs/cov_ens_smoke_%j.out
#SBATCH -e logs/cov_ens_smoke_%j.err
# Not dine2: results/analysis is a symlink into /cosma5, which dine2 does not mount
# (ROADMAP 2026-09-24). Memory sized from past MaxRSS instead of a whole node.

# Fast plumbing smoke test for coverage_ensemble_chain.py's new
# --phi_mass_matrix / --no_sample_cl_phiphi arguments. Tiny lmax, a handful of
# sweeps -- this validates the argument plumbing, the run_gibbs_chain return
# ARITY (5-tuple with Block 4 off vs 6-tuple with it on; a hardcoded unpack
# here previously crashed a completed 3000-sweep chain, achievements.md) and
# the npz write path. It says nothing about equilibration and is not a gate.
#
# The real ensemble array is submitted with --dependency=afterok on this job,
# so a plumbing failure cannot burn 12 production tasks.
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

mkdir -p logs results/analysis/coverage_ensemble_smoke
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-smoke"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0

set -e

echo "=== Smoke 1/2: Block 4 OFF + block mass matrix (the production config) ==="
$PYTHON -u scripts/coverage_ensemble_chain.py \
    --realization 0 --lmax 16 --nside 16 \
    --n_burnin 2 --n_samples 4 --n_lfs 3 --phi_n_lfs 4 \
    --phi_mass_matrix block --no_sample_cl_phiphi \
    --outdir results/analysis/coverage_ensemble_smoke \
    --checkpoint_every 2

echo "=== Smoke 2/2: Block 4 ON (regression: the pre-existing 6-tuple path) ==="
$PYTHON -u scripts/coverage_ensemble_chain.py \
    --realization 1 --lmax 16 --nside 16 \
    --n_burnin 2 --n_samples 4 --n_lfs 3 --phi_n_lfs 4 \
    --phi_mass_matrix prior \
    --outdir results/analysis/coverage_ensemble_smoke \
    --checkpoint_every 2

echo "=== Smoke 3/3: aggregator must tolerate the mixed/absent cl_phiphi key ==="
$PYTHON -u scripts/aggregate_coverage_ranks.py \
    --indir results/analysis/coverage_ensemble_smoke \
    --thin 1 \
    --out results/analysis/coverage_ensemble_smoke/agg_smoke.npz

echo "SMOKE OK"
