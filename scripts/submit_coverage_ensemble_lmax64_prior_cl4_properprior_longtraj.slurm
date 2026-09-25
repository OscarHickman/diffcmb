#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=13:00:00
#SBATCH -J cov_ens_l64_properprior_longtraj
#SBATCH -o logs/cov_ens_l64_properprior_longtraj_%A_%a.out
#SBATCH -e logs/cov_ens_l64_properprior_longtraj_%A_%a.err
#SBATCH --exclusive
#SBATCH --array=0-11

# ===================================================================
# 2026-09-01: DOES A LONGER phi TRAJECTORY CLEAR OPTION 2's STRICT SBC RANK?
#
# THE QUESTION. Job 11903182 (proper prior nu=6, corrected inverse-Gamma
# shape) improved the strict C_L^phiphi SBC rank from 0.25-0.28 (KS_p=0)
# to mean_u = 0.3802, KS_p = 0.0013 -- a large, real gain from the dof fix,
# but it does not clear. Everything measured since points AWAY from the
# conditional and TOWARD mixing:
#
#   - Block 4 is exact given phi. Its PIT is mean_u = 0.4999, KS_p = 0.42,
#     and as of 2026-09-01 that pass is no longer vacuous: the misalignment
#     control now runs at several lags, and while lag-1 passes (phi
#     autocorrelation 0.915), lag-10 (0.522) and lag-50 (0.086) are both
#     rejected at KS_p = 0. The test has demonstrated power.
#   - The alm rank is clean (0.5052, KS_p = 0.41).
#   - The phi field rank (0.4115) and the C_L^phiphi rank are lowest in the
#     SAME bin, [10,30). Since Block 4 is exact given phi, a low C_L^phiphi
#     rank means its conditioning S_L(phi) is high -- and measured phi
#     power/truth is 1.08-1.14. The deficit is in Block 3 or its mixing.
#   - It is not burn-in: the strict rank is flat at 0.400 on the 2nd half
#     and 0.396 on the last quarter, and phi power/truth RISES slightly
#     across the chain (1.002 -> 1.060).
#   - Block 4 ON mixes far worse on the same phi block: tau_int median
#     24-56 per bin (max 92.7) vs 4.7-27 (max 42.5) with Block 4 off;
#     split-Rhat max 1.64 vs 1.31. This is the (phi, C_L^phiphi) funnel
#     already on record in lensing.py's ancillary-move note.
#
# THE TEST. One variable: phi_n_lfs 240 -> 480. Per achievements.md the
# phi-block trajectory length is the parameter that actually controls phi
# mixing, and the phi acceptance rate on 11903182 was 0.82 -- high, so the
# step size is not the binding constraint and lengthening the trajectory is
# the right knob. EVERYTHING else is identical to job 11903182: lmax=64,
# nside=64, phi_mass_matrix='prior', nu=6, n_burnin=100, 600 samples,
# n_lfs=10, MAP start, same per-realization seeds, so the truth triples are
# bit-identical and the two ensembles are directly comparable.
#
# READING THE RESULT.
#   - Strict C_L^phiphi rank moves toward 0.5 with tau_int down => the
#     residual was non-convergence under the funnel. Option 2 becomes
#     quotable and the differentiator figure loses its caveat.
#   - Rank stays ~0.38 while tau_int halves => mixing is NOT the cause and
#     suspicion moves to Block 3 (phi | alm, C_l) itself. That is a
#     different investigation, and this run is what licenses it.
#   Either way the answer is informative; there is no null outcome.
#
# WALLTIME 13h, NOT 24h -- deliberate, 2026-09-01. A cluster-wide MAINT
# reservation (root_321) starts 2026-09-02T15:00 and covers all of dine2
# (gc[001-008]), so SLURM refuses to start any job whose walltime would run
# into it: the first submission of this script (job 11912084, 24h) sat at
# ReqNodeNotAvail and would not have started at all. 13h fits the ~19h window
# and is ~1.35x the ~9.6h estimate. dine2 has 8 nodes and this array has 12
# exclusive tasks, so tasks 8-11 form a second wave that starts only as the
# first finishes and will likely be cut by the reservation -- that is expected
# and recoverable: checkpoint_every=50 writes state and run_gibbs_chain resumes
# automatically from an existing checkpoint, so resubmitting after maintenance
# continues those chains rather than restarting them. Realizations are
# independent, so an 8-chain aggregate is a valid (lower-N) result meanwhile.
#
# COST. The phi block dominates, so doubling its trajectory roughly doubles
# the ~4.8h/realization to ~9.6h -- still comfortably inside the 24h
# walltime, and checkpoint_every=50 allows a resume if it is not.
#
# Job cap: 12 array tasks. Re-check before submitting (limit is 200):
#   squeue -u dc-hick2 --noheader -r -o "%a" | grep -c durham
#
# HARVEST. .err for tracebacks first (SLURM COMPLETED is not enough), then
# per-realization phi power ratio O(1), then re-derive --thin from THIS
# run's own tau_int (do not carry 90 over) and:
#   PYTHONPATH=diffcmb .venv/bin/python scripts/validate_coverage_rank_nulls.py \
#       --indir results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_longtraj \
#       --thin <tau> --control_lags 1,10,50
#   PYTHONPATH=diffcmb .venv/bin/python scripts/aggregate_coverage_ranks.py \
#       --indir results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_longtraj \
#       --thin <tau>
# The decisive line is STRICT C_L^phiphi SBC; the aggregator's C_L^phiphi
# row is interval coverage only and needs its null. Check the PIT verdict
# line says a control was rejected before reporting the aligned pass.
# ===================================================================

LMAX=64
NSIDE=64
BURNIN=100
SAMPLES=600
N_LFS=10
PHI_N_LFS=480

# results/analysis is a symlink into /cosma5, which dine2 does not mount (ROADMAP
# 2026-09-24). Fail loudly here rather than at the first write.
if [ ! -d /cosma/apps/durham/dc-hick2/diffcmb/results/analysis/ ]; then
    echo "results/analysis (/cosma5) is not mounted on $(hostname); relocate outputs before running on dine2" >&2
    exit 1
fi
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

mkdir -p logs results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_longtraj
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "Coverage ensemble realization ${SLURM_ARRAY_TASK_ID} at lmax=${LMAX} (prior mass matrix, Block 4 ON, proper prior nu=6, phi_n_lfs=${PHI_N_LFS} = 2x job 11903182)"
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
    --outdir results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_longtraj \
    --checkpoint_every 50

rc=$?
echo "Realization ${SLURM_ARRAY_TASK_ID} exited with status $rc"
# Propagate the status so sacct shows FAILED rather than COMPLETED.
exit $rc
