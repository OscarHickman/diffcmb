#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J coverage_ensemble
#SBATCH -o logs/coverage_ensemble_%A_%a.out
#SBATCH -e logs/coverage_ensemble_%A_%a.err
#SBATCH --exclusive
#SBATCH --array=0-15

# ROADMAP.md Section 1, Priority 1: the multi-realization rank/coverage
# ensemble -- THE headline validation. One array task per realization; then
#   PYTHONPATH=diffcmb .venv/bin/python scripts/aggregate_coverage_ranks.py \
#       --indir results/analysis/coverage_ensemble --thin <from pilot>
#
# *** DO NOT SUBMIT UNTIL BOTH OF THESE HOLD ***
#
# 1. scripts/pilot_coverage_equilibration.py has returned GO at this
#    (lmax, phi_n_lfs, n_samples) configuration. A rank test on chains that have
#    not equilibrated produces confidently wrong uniformity plots -- strictly
#    worse than no test, and this project has been burned by exactly that twice
#    (achievements.md). The pilot is the gate; this is the spend.
# 2. --thin for the aggregator has been set from the pilot's measured lag-k
#    autocorrelation. Unthinned posterior draws are autocorrelated, so their
#    ranks are not draws from the rank's null distribution and the uniformity
#    test is invalid regardless of how many chains ran.
#
# Also read scripts/aggregate_coverage_ranks.py's header before quoting any
# number from this: the field-level (alm, phi) ranks are the calibration
# evidence, while the spectrum-level (C_l, C_L^phiphi) rows are interval
# coverage only -- Blocks 1 and 4 have improper flat implied priors, so no
# strict SBC rank exists for the spectra. That distinction has to survive into
# the figure caption.
#
# Cost: the pilot measured ~29s/sweep at lmax=128/nside=128/phi_n_lfs=80, so
# 700 sweeps ~ 5.6h per realization -- comfortably one 24h walltime each, no
# resubmission expected. Each task checkpoints anyway. 16 tasks + the ~15 other
# durham jobs sits far under the 200-job cap, but re-check before submitting:
#   squeue -u dc-hick2 --noheader -r -o "%a" | grep -c durham
LMAX=128
NSIDE=128
BURNIN=100
SAMPLES=600
N_LFS=10
PHI_N_LFS=80

# results/analysis is a symlink into /cosma5, which dine2 does not mount (ROADMAP
# 2026-09-24). Fail loudly here rather than at the first write.
if [ ! -d /cosma/apps/durham/dc-hick2/diffcmb/results/analysis/ ]; then
    echo "results/analysis (/cosma5) is not mounted on $(hostname); relocate outputs before running on dine2" >&2
    exit 1
fi
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

mkdir -p logs results/analysis/coverage_ensemble
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "Coverage ensemble realization ${SLURM_ARRAY_TASK_ID} at lmax=${LMAX}"
echo "Using Python: $($PYTHON --version 2>&1)"

$PYTHON -u scripts/coverage_ensemble_chain.py \
    --realization ${SLURM_ARRAY_TASK_ID} \
    --lmax $LMAX \
    --nside $NSIDE \
    --n_burnin $BURNIN \
    --n_samples $SAMPLES \
    --n_lfs $N_LFS \
    --phi_n_lfs $PHI_N_LFS \
    --outdir results/analysis/coverage_ensemble \
    --checkpoint_every 50
