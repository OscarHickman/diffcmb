#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J cov_ens_l64_prior_cl4_mapfix
#SBATCH -o logs/cov_ens_l64_prior_cl4_mapfix_%A_%a.out
#SBATCH -e logs/cov_ens_l64_prior_cl4_mapfix_%A_%a.err
#SBATCH --exclusive
#SBATCH --array=0-11

# ===================================================================
# 2026-08-30 FULL 12-CHAIN RERUN, POST-MAP-START-FIX.
#
# The prior ensemble run at this configuration (job 11887897, output in
# results/analysis/coverage_ensemble_lmax64_prior_cl4/) failed completely:
# phi and C_L^phiphi ranked 0/8 in every ell-bin of all 12 realizations,
# frozen 1e3-1e5x above the true power. Root cause (ROADMAP.md item 7,
# 2026-08-28): coverage_ensemble_chain.py cold-started alm from a prior
# draw instead of running the MAP pre-solve pilot_coverage_equilibration.py
# uses, so the phi block had ~1000x too much residual to absorb and the
# Block-4-refitted prior is exactly scale-free in phi amplitude (no
# restoring force). Fixed in coverage_ensemble_chain.py (MAP start,
# map_steps=2000 default, n_burnin=400 default).
#
# Single-realization validation (job 11892308, this same prior_cl4 config
# but n_burnin=100) PASSED cleanly (checked 2026-08-30):
#   - phi power / truth power ratio: 1.40-1.65 across the chain (healthy
#     pilots land in 0.09-2.5; the broken ensemble was 1e3-1e5)
#   - alm-vs-truth cosine similarity: flat at 0.9998 for the whole chain
#     (broken run: 0.24-0.91 and decreasing)
#   - logp: plateaued (linear-fit slopes -0.20 / +0.07 over 300-sample
#     halves, noise-level against a std of ~52; broken run drifted
#     monotonically with no plateau)
# n_burnin=100 (not the new 400 default) was sufficient here because the
# MAP start alone already gives alm cosine 0.9999 -- confirmed by this
# validation run, so this rerun keeps n_burnin=100 to match it exactly
# rather than introducing an unvalidated change.
#
# This is the SAME (lmax=64, phi_mass_matrix='prior', Block 4 ON)
# configuration as job 11849969 / the original prior_cl4 ensemble -- see
# submit_coverage_ensemble_lmax64_prior_cl4.slurm for the full mass-matrix
# provenance (why 'prior' not 'block': job 11874976 gated 'block' NO-GO
# post-ordering-fix, 0/4 bins). Only the alm-init pipeline changed.
#
# Do NOT write into results/analysis/coverage_ensemble_lmax64/ (pre-fix,
# job 11848757) or results/analysis/coverage_ensemble_lmax64_prior_cl4/
# (cold-start bug, job 11887897) -- both are invalid. Fresh directory here.
#
# Cost: ~24.8s/sweep at this configuration (job 11849969) x 700 sweeps
# (100 burn-in + 600 samples) = ~4.8h/realization, well inside 24h walltime.
#
# Job cap: 12 array tasks. Re-check before submitting (limit is 200):
#   squeue -u dc-hick2 --noheader -r -o "%a" | grep -c durham
#
# After all tasks finish, aggregate with --thin set from the two clean
# bins' tau_int (70-78, rounded up), and report the [60,64) bin's known
# weak tau_int=326 separately rather than folding it into the aggregate:
#   PYTHONPATH=diffcmb .venv/bin/python scripts/aggregate_coverage_ranks.py \
#       --indir results/analysis/coverage_ensemble_lmax64_prior_cl4_mapfix --thin 90
# ===================================================================

LMAX=64
NSIDE=64
BURNIN=100
SAMPLES=600
N_LFS=10
PHI_N_LFS=240

# results/analysis is a symlink into /cosma5, which dine2 does not mount (ROADMAP
# 2026-09-24). Fail loudly here rather than at the first write.
if [ ! -d /cosma/apps/durham/dc-hick2/diffcmb/results/analysis/ ]; then
    echo "results/analysis (/cosma5) is not mounted on $(hostname); relocate outputs before running on dine2" >&2
    exit 1
fi
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

mkdir -p logs results/analysis/coverage_ensemble_lmax64_prior_cl4_mapfix
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "Coverage ensemble realization ${SLURM_ARRAY_TASK_ID} at lmax=${LMAX} (prior mass matrix, Block 4 ON, MAP-start fix)"
echo "Using Python: $($PYTHON --version 2>&1)"

$PYTHON -u scripts/coverage_ensemble_chain.py \
    --realization ${SLURM_ARRAY_TASK_ID} \
    --lmax $LMAX \
    --nside $NSIDE \
    --n_burnin $BURNIN \
    --n_samples $SAMPLES \
    --n_lfs $N_LFS \
    --phi_n_lfs $PHI_N_LFS \
    --phi_mass_matrix prior \
    --outdir results/analysis/coverage_ensemble_lmax64_prior_cl4_mapfix \
    --checkpoint_every 50

rc=$?
echo "Realization ${SLURM_ARRAY_TASK_ID} exited with status $rc"
# Propagate the status so sacct shows FAILED rather than COMPLETED.
exit $rc
