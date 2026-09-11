#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J cov_ens_l64_prior_cl4_properprior_packingv2_extendN
#SBATCH -o logs/cov_ens_l64_prior_cl4_properprior_packingv2_extendN_%A_%a.out
#SBATCH -e logs/cov_ens_l64_prior_cl4_properprior_packingv2_extendN_%A_%a.err
#SBATCH --exclusive
#SBATCH --array=12-23

# ===================================================================
# 2026-09-11 EXTEND THE BLOCK-4-ON PROPER-PRIOR PACKING-V2 ENSEMBLE
# (job 11965813) FROM N=12 TO N=24 REALIZATIONS.
#
# WHY, two independent reasons -- this run does double duty:
#
#  1. THE DIFFERENTIATOR FIGURE (ROADMAP.md Next actions #3, the paper's
#     headline object). scripts/plot_joint_cl_clpp_posterior.py measures the
#     within-posterior (C_l^TT, C_L^phiphi) correlation and is currently a
#     NULL at the available sample size: 1 of 16 bin-pair cells exceeds its
#     permutation null against 0.8 expected by chance, because at ~168 pooled
#     draws the estimator throws |r| ~ 0.15 on uncorrelated data -- the same
#     size as the effect. Resolving |r|=0.10 at 2 sigma needs ~2.4x that
#     ensemble. The error bar is a CHAIN-level bootstrap (only the 12 chains
#     are independent, not the 600 sweeps), so the lever is more CHAINS, not
#     longer ones -- doubling to 24 buys the sqrt(2) that matters.
#     Job 11903182's 12 chains CANNOT be pooled in: they are pre-restoration
#     2L packing, a different model (achievements.md). 11965813 + this run is
#     the largest correctly-packed Block-4-ON ensemble available.
#
#  2. THE [10,30) RESIDUAL. Job 11965813's strict C_L^phiphi SBC rank came in
#     at 0.4245 (KS_p=0.00395), still localised to [10,30) (0.2812, KS_p
#     0.0047) at N=12. Doubling N is the same small-N-vs-real-defect
#     discriminator that job 11965828 just applied to the phi [30,60) flag in
#     the Block-4-OFF ensemble (where the flag relaxed and the defect reading
#     was withdrawn). This does NOT touch phi mixing and is NOT a tuning run
#     -- the configuration is byte-identical to 11965813 -- so it is outside
#     ROADMAP.md's "no further phi-equilibration tuning without sign-off"
#     rule. It buys evidence on the open question without pre-judging it.
#
# SAME outdir as job 11965813
# (results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_packingv2/).
# coverage_ensemble_chain.py seeds every stream off --realization from
# disjoint 1e6-spaced blocks (_STREAM_TRUTH/_START/_NOISE/_CLPP_PRIOR), so
# realizations 12-23 are new independent draws that APPEND 12 more
# chain_rXXX.npz alongside the existing 12 -- nothing is overwritten.
# Harvesting reads all 24 together.
#
# IDENTICAL configuration to 11965813: lmax=64, nside=64,
# phi_mass_matrix='prior', n_burnin=100, 600 samples, n_lfs=10,
# phi_n_lfs=240, cl_phiphi_prior_nu=6.0, restored 2L+1 packing. ~5h per
# realization inside the 24h walltime; checkpoint_every=50 allows resume.
#
# Job cap: 12 array tasks (limit is 200). Re-check before submitting:
#   squeue -u dc-hick2 --noheader -r -o "%a" | grep -c durham
#
# HARVEST. .err per task for tracebacks first (SLURM COMPLETED is not
# enough), confirm each realization's phi power ratio is O(1), then re-run
# BOTH harvest scripts over all 24 (same outdir covers both batches):
#   PYTHONPATH=diffcmb .venv/bin/python scripts/aggregate_coverage_ranks.py \
#       --indir results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_packingv2 \
#       --thin 90
#   PYTHONPATH=diffcmb .venv/bin/python scripts/validate_coverage_rank_nulls.py \
#       --indir results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_packingv2 \
#       --thin 90 --control_lags 1,10,50
# Compare the strict C_L^phiphi rank and the [10,30) bin at N=24 against the
# N=12 pair (0.4245 / KS_p 0.00395; bin 0.2812 / KS_p 0.0047): relaxing =
# small-N, sharpening = a real localised defect. Check the PIT verdict line
# reports a rejected control (lag-1 alone is vacuous here, phi lag-1
# decorrelation is +0.915) before quoting the aligned pass. Then rebuild the
# differentiator figure over all 24 chains:
#   PYTHONPATH=diffcmb .venv/bin/python scripts/plot_joint_cl_clpp_posterior.py
# ===================================================================

LMAX=64
NSIDE=64
BURNIN=100
SAMPLES=600
N_LFS=10
PHI_N_LFS=240

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

mkdir -p logs results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_packingv2
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "Coverage ensemble realization ${SLURM_ARRAY_TASK_ID} at lmax=${LMAX} (prior mass matrix, Block 4 ON + PROPER C_L^phiphi prior nu=6, restored 2L+1 packing)"
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
    --cl_phiphi_prior_nu 6.0 \
    --outdir results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_packingv2 \
    --checkpoint_every 50

echo "Realization ${SLURM_ARRAY_TASK_ID} exited with status $?"
