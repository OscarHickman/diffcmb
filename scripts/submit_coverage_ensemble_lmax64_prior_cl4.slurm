#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J cov_ens_l64_prior_cl4
#SBATCH -o logs/cov_ens_l64_prior_cl4_%A_%a.out
#SBATCH -e logs/cov_ens_l64_prior_cl4_%A_%a.err
#SBATCH --exclusive
#SBATCH --array=0-11

# ROADMAP.md Section 1, Priority 1 -- THE headline validation, launched at
# the post-ordering-fix configuration the user chose to ship on 2026-08-27
# (achievements.md), after the production `block` mass matrix was gated
# NO-GO across all 4 l-bins post-fix (job 11874976).
#
# Configuration provenance:
#   lmax=64, nside=64, phi_n_lfs=240, phi_mass_matrix='prior' (diagonal, no
#   Nystrom correction), Block 4 ON (sample_cl_phiphi=True, the default) ==
#   job 11849969, harvested 2026-08-25, re-scored 2026-08-27
#   (scripts/rescore_phi_equilibration.py, post-bugfix): 2 of 4 l-bins
#   cleanly pass (Rhat 1.00-1.03, ESS 42-47), [2,10) is marginal (Rhat right
#   at the 1.05 boundary), and [60,64) is a genuine long-lived mode (Rhat
#   1.404, tau_int 326, ESS 10). User decision: ship anyway, accepting
#   [60,64) as a known weak bin rather than excluding it or debugging
#   further -- report it separately in any coverage figure, don't fold it
#   into an aggregate thin value that hides it.
#
# Why not `phi_mass_matrix='block'`: job 11874976 gated that exact
# configuration post-fix on 2026-08-27 and it was worse -- 0/4 bins pass
# (achievements.md). The Nystrom cross-L correction that fixed equilibration
# pre-fix, Block-4-off (job 11781626) now regresses mixing once Block 4 is
# active, post-fix. Root cause not yet understood; not blocking this launch
# per the user's explicit sign-off.
#
# Do NOT write into results/analysis/coverage_ensemble_lmax64/ -- that
# directory holds the invalid pre-fix chains from cancelled job 11848757
# (tasks 0-5 completed against the broken alm-ordering path before
# cancellation). This uses a fresh output directory.
#
# Cost: job 11849969 measured ~24.8s/sweep at this exact configuration
# (lmax=64, phi_n_lfs=240, Block 4 ON) over 3700 sweeps. At 700 sweeps
# (100 burn-in + 600 samples, matching the existing lmax=64 ensemble design)
# that's ~4.8h per realization -- comfortably inside one 24h walltime. Each
# task checkpoints every 50 sweeps regardless.
#
# Job cap: 12 array tasks. Re-check before submitting (limit is 200):
#   squeue -u dc-hick2 --noheader -r -o "%a" | grep -c durham
#
# After all tasks finish, aggregate with --thin set from the two clean bins'
# tau_int (70-78, rounded up):
#   PYTHONPATH=diffcmb .venv/bin/python scripts/aggregate_coverage_ranks.py \
#       --indir results/analysis/coverage_ensemble_lmax64_prior_cl4 --thin 90
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

mkdir -p logs results/analysis/coverage_ensemble_lmax64_prior_cl4
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "Coverage ensemble realization ${SLURM_ARRAY_TASK_ID} at lmax=${LMAX} (prior mass matrix, Block 4 ON)"
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
    --outdir results/analysis/coverage_ensemble_lmax64_prior_cl4 \
    --checkpoint_every 50

rc=$?
echo "Realization ${SLURM_ARRAY_TASK_ID} exited with status $rc"
# Propagate the status so sacct shows FAILED rather than COMPLETED.
exit $rc
