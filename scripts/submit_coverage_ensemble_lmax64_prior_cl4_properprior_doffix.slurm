#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J cov_ens_l64_prior_cl4_properprior_doffix
#SBATCH -o logs/cov_ens_l64_prior_cl4_properprior_doffix_%A_%a.out
#SBATCH -e logs/cov_ens_l64_prior_cl4_properprior_doffix_%A_%a.err
#SBATCH --exclusive
#SBATCH --array=0-11

# ===================================================================
# 2026-08-31 RERUN AGAINST THE CORRECTED INVERSE-GAMMA SHAPE PARAMETER.
#
# The predecessor of this job ran with Blocks 1 and 4 using alpha = L - 0.5,
# which assumes 2L+1 real dof per multipole. The packed vector actually
# carries 2L (splittosingularalm forces Im(a_{L,1}) = 0), so the exact
# conditional is InvGamma(k_L/2 + a0, b0 + S_L/2) -- alpha = L - 1 under the
# flat prior. E[C_L] was biased by (L-1.5)/(L-2): 0.8% at L=63, 6% at L=10,
# 50% at L=3, undefined at L=2. Now derived from the packing itself
# (alm_utils.invgamma_shape_for_spectrum) so it tracks the layout.
#
# Rerunning BOTH configurations rather than only the proper-prior one: the
# Block-4-OFF run still uses Block 1, so its C_l target moved too, and its
# headline phi rank should not rest on a since-fixed bug.
#
# The decisive check on the proper-prior run is the STRICT C_L^phiphi SBC
# rank (its truth is drawn from the prior the sampler targets): it was
# mean_u = 0.25-0.28, KS_p = 0.0000 before the fix and must come back
# uniform now. That statistic is what exposed the bug in the first place.
#
# NOTE for the flat-prior/Block-4-OFF variant: the corrected shape is
# heavier-tailed at low L (alpha = 1 at L=2, so the conditional mean does not
# exist). Watch the L=2 behaviour when harvesting -- if C_L^phiphi or C_l
# excursions at L=2 destabilise the chain, that is the improper prior biting
# and is an argument for the proper-prior configuration, not a regression.
# ===================================================================
# ===================================================================
# 2026-08-31 OPTION 2: coverage ensemble with a PROPER prior on C_L^phiphi.
#
# WHY. With Block 4 on and the default flat improper prior on C_L^phiphi,
# integrating C_L out leaves a phi marginal proportional to S_L^{-(L-0.5)},
# which against the (2L+1)-component measure r^{2L}dr is p(r) ∝ r^1 --
# FLAT IN S_L, improper and rising with amplitude. Consequences already
# recorded: the 2026-08-28 ensemble froze phi 1e3-1e5x above truth with no
# restoring force (job 11887897), and the 2026-08-31 ensemble's phi rank
# (mean_u=0.367, job 11899585) cannot be read as calibration evidence
# because the truth came from a proper N(0,C_fid) the sampler is not
# targeting.
#
# Option 1 (job 11900600, Block 4 OFF) removes the problem by removing
# Block 4 -- but Block 4 is what produces C_L^phiphi, and the joint
# (C_l, C_L^phiphi) posterior IS this paper's headline differentiator.
# This run is the version that keeps the differentiator AND has a proper
# target: a conjugate InvGamma(nu/2, nu*C_L^fid/2) prior on each C_L, so
#
#   C_L^phiphi | phi ~ InvGamma(L - 0.5 + nu/2, (S_L + nu*C_L^fid)/2)
#
# and the phi marginal tail becomes r^{1-nu} -- normalisable for nu > 2
# (enforced in lensing.py; nu=2 still diverges logarithmically).
#
# WHY nu=6. nu reads as an effective number of prior pseudo-modes, on the
# same footing as the 2L+1 real modes the data supplies at multipole L. At
# L=2 (5 modes) nu=6 genuinely regularises, which is exactly where the
# improper prior does its damage; by L=63 (127 modes) it is ~5% of the
# information and the prior is negligible. alpha_0 = nu/2 = 3 also puts the
# tail at r^{-5}, comfortably proper with finite variance, rather than
# sitting near the nu=2 boundary.
#
# TRUTH IS DRAWN FROM THE SAME PRIOR. coverage_ensemble_chain.py, when
# --cl_phiphi_prior_nu is set, draws C_L^phiphi ~ InvGamma(nu/2,
# nu*C_L^fid/2) first and then phi_true ~ N(0, C_L). Generative process and
# sampler target therefore agree, which is what makes the phi rank in THIS
# run a valid strict SBC statistic -- unlike job 11899585. The chain itself
# is still handed the FIDUCIAL spectrum (the prior centre), not the truth
# draw. C_l^TT is untouched and its row remains interval coverage only, so
# it still needs scripts/validate_coverage_rank_nulls.py before any flag on
# that row is read as bias.
#
# Otherwise identical to job 11899585: lmax=64, nside=64,
# phi_mass_matrix='prior', n_burnin=100, 600 samples, n_lfs=10,
# phi_n_lfs=240, MAP start. Cost ~4.8h/realization inside 24h walltime.
#
# Job cap: 12 array tasks. Re-check before submitting (limit is 200):
#   squeue -u dc-hick2 --noheader -r -o "%a" | grep -c durham
#
# HARVEST. Check .err for tracebacks (SLURM COMPLETED is not enough), check
# per-realization phi power ratio is O(1), then aggregate and compare the
# phi row against 0.5 DIRECTLY (it is a real SBC rank here):
#   PYTHONPATH=diffcmb .venv/bin/python scripts/aggregate_coverage_ranks.py \
#       --indir results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_doffix --thin 90
# Re-derive --thin from this run's own tau_int. For the two spectrum rows,
# get the null from scripts/validate_coverage_rank_nulls.py first.
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

mkdir -p logs results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_doffix
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "Coverage ensemble realization ${SLURM_ARRAY_TASK_ID} at lmax=${LMAX} (prior mass matrix, Block 4 ON + PROPER C_L^phiphi prior nu=6, valid SBC)"
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
    --outdir results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_doffix \
    --checkpoint_every 50

rc=$?
echo "Realization ${SLURM_ARRAY_TASK_ID} exited with status $rc"
# Propagate the status so sacct shows FAILED rather than COMPLETED.
exit $rc
