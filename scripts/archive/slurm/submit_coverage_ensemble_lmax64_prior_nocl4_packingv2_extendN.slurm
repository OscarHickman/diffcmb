#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J cov_ens_l64_prior_nocl4_packingv2_extendN
#SBATCH -o logs/cov_ens_l64_prior_nocl4_packingv2_extendN_%A_%a.out
#SBATCH -e logs/cov_ens_l64_prior_nocl4_packingv2_extendN_%A_%a.err
#SBATCH --exclusive
#SBATCH --array=12-23

# ===================================================================
# 2026-09-09 EXTEND THE HEADLINE PACKING-V2 ENSEMBLE (job 11955622) FROM
# N=12 TO N=24 REALIZATIONS, TO CHASE THE FLAGGED phi [30,60) BIN.
#
# WHY. ROADMAP.md's headline SBC result (job 11955622, 12 realizations,
# Block 4 OFF, restored 2L+1 packing) flags one bin: phi [30,60), KS_p=0.005
# at N=12 -- "read as small-N noise, unconfirmed either way" (also the only
# bin flagged pre-restoration). Next actions #2. Doubling N is the direct
# test: if the flag persists/sharpens at N=24 it graduates from noise to a
# real, if small, defect; if KS_p relaxes toward the other three bins it was
# noise.
#
# SAME outdir as job 11955622
# (results/analysis/coverage_ensemble_lmax64_prior_nocl4_packingv2/) --
# realizations 12-23 are NEW, disjoint seed streams (coverage_ensemble_chain.py
# seeds every stream off --realization), so this appends 12 more chain_rXXX.npz
# files to the existing 12 rather than duplicating or overwriting anything.
# Harvesting reads all 24 together.
#
# IDENTICAL configuration to 11955622: lmax=64, nside=64,
# phi_mass_matrix='prior', n_burnin=100, 600 samples, n_lfs=10,
# phi_n_lfs=240, Block 4 OFF (--no_sample_cl_phiphi). Restored 2L+1 packing
# (no code change since 11955622).
#
# Job cap: 12 array tasks. Re-check before submitting (limit is 200):
#   squeue -u dc-hick2 --noheader -r -o "%a" | grep -c durham
#
# HARVEST. Check .err per task first (SLURM COMPLETED alone is not enough),
# confirm phi_power_ratio_to_truth O(1) for realizations 12-23, then re-run
# the aggregator over ALL 24 realizations (same outdir covers both batches):
#   PYTHONPATH=diffcmb .venv/bin/python scripts/aggregate_coverage_ranks.py \
#       --indir results/analysis/coverage_ensemble_lmax64_prior_nocl4_packingv2 \
#       --thin 90
# Compare the phi [30,60) bin's KS_p at N=24 against the N=12 value (0.005)
# and the pooled headline mean_u/KS_p against the N=12 pair (phi 0.4792,
# alm 0.5130).
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

mkdir -p logs results/analysis/coverage_ensemble_lmax64_prior_nocl4_packingv2
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "Coverage ensemble realization ${SLURM_ARRAY_TASK_ID} at lmax=${LMAX} (prior mass matrix, Block 4 OFF, restored 2L+1 packing) -- N=12->24 extension"
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
    --no_sample_cl_phiphi \
    --outdir results/analysis/coverage_ensemble_lmax64_prior_nocl4_packingv2 \
    --checkpoint_every 50

rc=$?
echo "Realization ${SLURM_ARRAY_TASK_ID} exited with status $rc"
# Propagate the status so sacct shows FAILED rather than COMPLETED.
exit $rc
