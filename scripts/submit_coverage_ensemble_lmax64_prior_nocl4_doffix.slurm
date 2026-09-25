#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J cov_ens_l64_prior_nocl4_doffix
#SBATCH -o logs/cov_ens_l64_prior_nocl4_doffix_%A_%a.out
#SBATCH -e logs/cov_ens_l64_prior_nocl4_doffix_%A_%a.err
#SBATCH --exclusive
#SBATCH --array=0-11

# ===================================================================
# 2026-08-31 RERUN AGAINST THE CORRECTED INVERSE-GAMMA SHAPE PARAMETER.
#
# The predecessor (job 11900600) ran with Blocks 1 and 4 using alpha = L-0.5,
# which assumes 2L+1 real dof per multipole. The packed vector actually
# carries 2L (splittosingularalm forces Im(a_{L,1}) = 0), so the exact
# conditional is InvGamma(k_L/2 + a0, b0 + S_L/2) -- alpha = L-1 under the
# flat prior. E[C_l] was biased by (l-1.5)/(l-2): 0.8% at l=63, 6% at l=10,
# 50% at l=3, undefined at l=2. Now derived from the packing itself
# (alm_utils.invgamma_shape_for_spectrum) so it tracks the layout.
#
# Block 4 is OFF here, but Block 1 is not, so this run's C_l target moved
# too and its headline phi rank should not rest on a since-fixed bug.
# Predecessor result to reproduce: phi pooled mean_u=0.4534 (KS_p=0.165),
# alm 0.5367 (KS_p=0.326).
#
# WATCH L=2: the corrected shape is heavier-tailed at low l (alpha = l-1 = 1
# at l=2, so the conditional mean does not exist). If l=2 excursions
# destabilise the chain that is the improper flat prior biting -- an argument
# for the proper-prior configuration, not a regression.
# ===================================================================

# ===================================================================
# 2026-08-31 OPTION 1: coverage ensemble with BLOCK 4 OFF.
#
# WHY. The Block-4-ON ensemble (job 11899585) came back healthy and every
# testable claim cleared (ROADMAP.md): the C_l^TT / C_L^phiphi rank FLAGs
# are artifacts of the rank statistic (it ranks the truth against its own
# conditional's mode), and Block 4 itself is exact on production chains
# (u = CDF_InvGamma(L-0.5, S_L(phi_i)/2)[C_L,i] uniform to KS_p=0.53,
# N=17856). What did NOT clear is the phi field rank: mean_u=0.367,
# pooled KS_p=0.0040.
#
# That residual is a TARGET-DEFINITION issue, not a sampler bug. With
# Block 4 on, the implied prior on C_L^phiphi is flat and improper;
# integrating it out leaves a phi marginal proportional to S_L^{-(L-0.5)},
# which against the (2L+1)-component radial measure r^{2L}dr is
# p(r) ∝ r^1 -- i.e. FLAT IN S_L, improper and rising with amplitude.
# So the phi amplitude is constrained by the likelihood alone, while
# phi_true was drawn from a PROPER N(0, C_L^phiphi,fid). Ranking a truth
# from a proper prior against a target with an improper one is not a valid
# calibration test, so the 0.367 cannot be read either way.
#
# Turning Block 4 OFF pins cl_phiphi_full at the same fiducial spectrum
# that generated phi_true (coverage_ensemble_chain.py line ~250). The phi
# prior is then proper and IDENTICAL to the generative process, so the phi
# SBC rank null IS uniform and the rank becomes a real exactness test --
# the strongest such evidence this project can produce. That is the point
# of this run.
#
# Trade-off, stated plainly: no C_L^phiphi samples, so this run does NOT
# produce the paper's headline joint (C_l, C_L^phiphi) figure. It is the
# exactness evidence; the differentiator figure comes from the Block-4-ON
# runs (and from the proper-prior work, Option 2).
#
# DELIBERATELY A ONE-VARIABLE CHANGE from job 11899585: same lmax=64,
# nside=64, phi_mass_matrix='prior', n_burnin=100, 600 samples, n_lfs=10,
# phi_n_lfs=240, same MAP start. ONLY --no_sample_cl_phiphi differs, so any
# difference in the phi rank is attributable to Block 4 and nothing else.
# (Note 'prior' not 'block' even though the historical lmax=64 GO was
# 'block'+Block-4-off: that GO predates the 2026-08-24 alm-ordering fix and
# is invalid. Post-fix, 'block' gated NO-GO at 0/4 bins, job 11874976.
# Holding the mass matrix fixed at the validated 'prior' keeps this a
# single-variable A/B.)
#
# Cost: ~24.8s/sweep x 700 sweeps = ~4.8h/realization (Block 4 off is if
# anything slightly cheaper), well inside 24h walltime.
#
# Job cap: 12 array tasks. Re-check before submitting (limit is 200):
#   squeue -u dc-hick2 --noheader -r -o "%a" | grep -c durham
#
# HARVEST. Check .err for tracebacks (SLURM COMPLETED is not enough -- see
# achievements.md's return-tuple-arity entry), confirm per-realization phi
# power ratio is O(1), then:
#   PYTHONPATH=diffcmb .venv/bin/python scripts/aggregate_coverage_ranks.py \
#       --indir results/analysis/coverage_ensemble_lmax64_prior_nocl4_doffix --thin 90
# Re-derive --thin from this run's own tau_int rather than assuming 90
# carries over from the Block-4-ON run. The cl_phiphi row will be absent by
# construction; the C_l^TT row still needs its null (scripts/
# validate_coverage_rank_nulls.py) before any flag is read as bias. The
# phi_power row is the one that is now a genuine SBC test -- read it
# against 0.5 directly.
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

mkdir -p logs results/analysis/coverage_ensemble_lmax64_prior_nocl4_doffix
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "Coverage ensemble realization ${SLURM_ARRAY_TASK_ID} at lmax=${LMAX} (prior mass matrix, Block 4 OFF -- proper phi prior, valid SBC)"
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
    --outdir results/analysis/coverage_ensemble_lmax64_prior_nocl4_doffix \
    --checkpoint_every 50

rc=$?
echo "Realization ${SLURM_ARRAY_TASK_ID} exited with status $rc"
# Propagate the status so sacct shows FAILED rather than COMPLETED.
exit $rc
