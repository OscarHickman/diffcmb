#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH -J pilot_block_n24_cl4_properprior
#SBATCH -o logs/pilot_block_n24_cl4_properprior_%j.out
#SBATCH -e logs/pilot_block_n24_cl4_properprior_%j.err
#SBATCH --exclusive

# ===================================================================
# 2026-09-02: DOES A HIGHER-RANK NYSTROM CORRECTION FIX [10,30) MIXING?
#
# THE QUESTION. Job 11913268 (cross-L Hessian coupling diagnostic) confirmed
# strong, reproducible coupling into the now-failing bin [10,30) -- receiving
# curvature from [2,10), projecting it outward to [30,60)/[60,64) -- across
# all 6 realizations tested. That is necessary but not sufficient to explain
# why phi_mass_matrix='block' (the Nystrom cross-L correction built for
# exactly this) already lost decisively to 'prior' with Block 4 ON (job
# 11874976, 0/4 bins pass, achievements.md). The untested, leading
# explanation (achievements.md, 2026-08-23) is a fixed-rank artifact:
# estimate_phi_block_hessian uses a CONSTANT n_probes=6 for the Nystrom
# approximation regardless of block size, and [2,10) -- the main driver of
# coupling into [10,30) -- lives entirely in the m=0..9 blocks, which are the
# LARGEST (K ~ lmax - m + 1 ~ 55-63 at lmax=64) and therefore worst
# approximated at fixed rank.
#
# THE TEST. One variable relative to job 11874976's config: phi_block_n_probes
# 6 -> 24 (4x the Nystrom rank in every block). Everything else matches
# TODAY's production configuration (job 11912088: lmax=64, nside=64,
# phi_n_lfs=480, proper prior nu=6, n_burnin=100, n_samples=600, MAP start),
# not job 11874976's flat-prior/phi_n_lfs=240 config, so this is a single
# realization directly comparable to 11912088's realization 0 chain. The
# extra probes only cost more gradient evals at the ONE-TIME burn-in
# curvature estimate (phi_fisher_warmup_iter, not every sweep), so 4x the
# rank is cheap relative to the full chain.
#
# READING THE RESULT. Compare this chain's [10,30) tau_int/split-Rhat/phi
# power ratio against:
#   - job 11912088 realization 0 (phi_mass_matrix='prior', same everything
#     else) -- the current production baseline.
#   - job 11874976 (phi_mass_matrix='block', n_probes=6, Block 4 ON, older
#     flat-prior config) -- the low-rank block failure.
# Clear improvement over 'prior' in [10,30) specifically => the
# rank-deficiency hypothesis was right; 'block' can be revived with a fixed
# rank and is worth a real ensemble. No improvement (or worse) => the earlier
# 'block' failure is not a rank artifact, and this route is closed more
# firmly than the single n_probes=6 test alone could establish -- fall back
# to investigating the likelihood/prior shape in [10,30) itself (e.g. Block
# 4's proper-prior construction at that L) rather than mass-matrix geometry.
#
# Job cap: 1 task. Re-check before submitting (limit is 200):
#   squeue -u dc-hick2 --noheader -r -o "%a" | grep -c durham
#
# WALLTIME 6h, NOT the ~10h a full 480-trajectory realization is estimated
# to need: the same cluster-wide MAINT reservation (root_321) from
# 2026-09-01's job 11912088 is still in effect, starting 2026-09-02T15:00 and
# covering all of dine2 -- a 10h request submitted at 08:15 sat at
# ReqNodeNotAvail (confirmed empirically) because it would run into the
# reservation. 6h fits the ~6h45m window before 15:00. If it doesn't finish,
# checkpoint_every=50 lets it resume automatically after maintenance clears --
# resubmit this same script unchanged and it will pick up the checkpoint.
# ===================================================================

LMAX=64
NSIDE=64
BURNIN=100
SAMPLES=600
N_LFS=10
PHI_N_LFS=480
PHI_BLOCK_N_PROBES=24

# results/analysis is a symlink into /cosma5, which dine2 does not mount (ROADMAP
# 2026-09-24). Fail loudly here rather than at the first write.
if [ ! -d /cosma/apps/durham/dc-hick2/diffcmb/results/analysis/ ]; then
    echo "results/analysis (/cosma5) is not mounted on $(hostname); relocate outputs before running on dine2" >&2
    exit 1
fi
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

mkdir -p logs results/analysis/pilot_block_n24_cl4_properprior
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "Single-realization pilot: phi_mass_matrix=block, phi_block_n_probes=${PHI_BLOCK_N_PROBES} (4x job 11874976's rank), Block 4 ON, proper prior nu=6, phi_n_lfs=${PHI_N_LFS} (matches job 11912088)"
echo "Using Python: $($PYTHON --version 2>&1)"

$PYTHON -u scripts/coverage_ensemble_chain.py \
    --realization 0 \
    --lmax $LMAX \
    --nside $NSIDE \
    --n_burnin $BURNIN \
    --n_samples $SAMPLES \
    --n_lfs $N_LFS \
    --phi_n_lfs $PHI_N_LFS \
    --phi_mass_matrix block \
    --phi_block_n_probes $PHI_BLOCK_N_PROBES \
    --cl_phiphi_prior_nu 6.0 \
    --outdir results/analysis/pilot_block_n24_cl4_properprior \
    --checkpoint_every 50

rc=$?
echo "Pilot exited with status $rc"
# Propagate the status so sacct shows FAILED rather than COMPLETED.
exit $rc
