#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J cov_ens_l64_prior_cl4_properprior_packingv2
#SBATCH -o logs/cov_ens_l64_prior_cl4_properprior_packingv2_%A_%a.out
#SBATCH -e logs/cov_ens_l64_prior_cl4_properprior_packingv2_%A_%a.err
#SBATCH --exclusive
#SBATCH --array=0-11

# ===================================================================
# 2026-09-09 RE-RUN OF JOB 11903182's CONFIG (Block 4 ON, proper C_L^phiphi
# prior, nu=6) UNDER THE RESTORED 2L+1 PACKING.
#
# WHY. ROADMAP.md's "Next actions" #1: every existing number for the
# Block-4-ON / proper-prior configuration (job 11903182: strict C_L^phiphi
# SBC rank 0.3802, KS_p=0.0013, residual concentrated in [10,30); job
# 11912088's phi_n_lfs=480 follow-up: 0.4196, KS_p=0.00049) predates the
# 2026-09-06 Im(a_{L,1}) dof restoration (achievements.md's "Open sampling
# question" section). Re-establish this configuration under the packing
# that is now the model before resuming any further Block 3 investigation
# (Hessian-coupling diagnosis, Nystrom mass matrix -- both closed/falsified
# under the OLD packing and not to be revisited without new evidence).
#
# The packing restoration's effect on Block 4's exactness PIT is already
# validated (achievements.md: 2L+1 dof re-validated at production scale,
# job 11955622 headline result, Block 4 OFF). This run tests whether the
# restoration also moves the Block-4-ON strict rank -- it changes both the
# alm dof (2L->2L+1, affecting Block 2/3's target) and Block 1/4's inverse-
# Gamma shape (L-1 -> L-0.5), either of which could shift a residual that
# was diagnosed (job 11913268) as cross-L Hessian curvature leaking into
# [10,30) from adjacent bins whose own dof just changed.
#
# DELIBERATELY the same configuration as job 11903182: lmax=64, nside=64,
# phi_mass_matrix='prior', n_burnin=100, 600 samples, n_lfs=10,
# phi_n_lfs=240, cl_phiphi_prior_nu=6.0. The packing restoration (629c0ee)
# is the only intended difference. Truth/data/start draws are not
# bit-identical to 11903182's even at matching --realization/seed, because
# hp.synalm now keeps the Im(a_{L,1}) it used to discard (same note as the
# packingv2 nocl4 and pilot scripts) -- expected, and the point of the run.
#
# Cost: ~4.8h/realization pre-restoration; packed vector grows by
# lmax-2=62 coords out of ~2080 (~3%), so budget ~5h/realization inside the
# 24h walltime. checkpoint_every=50 allows resume.
#
# Job cap: 12 array tasks. Re-check before submitting (limit is 200):
#   squeue -u dc-hick2 --noheader -r -o "%a" | grep -c durham
#
# HARVEST. Check .err for tracebacks per task first (SLURM COMPLETED alone
# is not enough -- achievements.md's return-tuple-arity entry), confirm
# each realization's phi_power_ratio_to_truth is O(1), re-derive --thin
# from this run's own tau_int (do not carry 90 over unchanged), then the
# decisive check is the STRICT C_L^phiphi SBC rank against 0.5:
#   PYTHONPATH=diffcmb .venv/bin/python scripts/validate_coverage_rank_nulls.py \
#       --indir results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_packingv2 \
#       --thin <tau> --control_lags 1,10,50
#   PYTHONPATH=diffcmb .venv/bin/python scripts/aggregate_coverage_ranks.py \
#       --indir results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_packingv2 \
#       --thin <tau>
# Compare the strict rank directly against job 11903182's 0.3802/0.0013
# baseline. Check the PIT verdict line reports a rejected control before
# reading the aligned Block-4-exactness pass as non-vacuous.
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

rc=$?
echo "Realization ${SLURM_ARRAY_TASK_ID} exited with status $rc"
# Propagate the status so sacct shows FAILED rather than COMPLETED.
exit $rc
