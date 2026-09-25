#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J pilot_packing_v2_l64_nocl4
#SBATCH -o logs/pilot_packing_v2_l64_nocl4_%A_%a.out
#SBATCH -e logs/pilot_packing_v2_l64_nocl4_%A_%a.err
#SBATCH --exclusive
#SBATCH --array=0-0

# ===================================================================
# 2026-09-06 SINGLE-REALIZATION VALIDATION OF THE Im(a_{L,1}) RESTORATION.
#
# The packing now carries 2L+1 real dof per multipole instead of 2L
# (achievements.md 2026-09-06; docs/notes/restore_missing_alm_dof_scoping.md).
# The unit suite is green (130 passed, 1 skipped) and an lmax=8 end-to-end
# smoke ran on the login node, but no production-scale chain has been run
# under the new layout. This is that chain: ONE realization before committing
# to the 12-chain re-run of the headline exactness result.
#
# Configuration is job 11903181 realization 0 EXACTLY -- lmax=64, nside=64,
# phi_n_lfs=240, phi_mass_matrix='prior', Block 4 OFF, n_burnin=100,
# n_samples=600, MAP start -- so the ONLY difference from the current
# headline ensemble (phi 0.4688 / alm 0.5312, both uniform) is the packing.
# Same --realization 0, so the seed streams and hence the truth/data/start
# are drawn from the same streams; note they are NOT bit-identical to
# 11903181's, because hp.synalm now keeps the Im(a_{L,1}) it used to discard.
#
# Block 4 OFF, not on, deliberately: that pins C_L^phiphi at the fiducial, so
# the phi prior is proper and identical to the process that generated the
# truth. It is the configuration the exactness claim rests on, so it is the
# one that has to be re-established first.
#
# PASS criteria, the same three the 2026-08-30 MAP-start validation used
# (job 11892308, achievements.md), read off the saved chain:
#   1. phi-power/truth ratio O(1)  -- healthy pilots land in 0.09-2.5; the
#      broken-ensemble signature is 1e3-1e5. The script itself raises after
#      saving if this fails, so a FAILED task with a chain_r000.npz on disk
#      means exactly this check tripped.
#   2. alm-vs-truth cosine similarity high and FLAT across the chain
#      (11892308: 0.9998 throughout; the broken run: 0.24-0.91 and falling).
#   3. logp plateaued -- fit slopes over the two halves at noise level, not
#      monotonic drift.
# NOT a pass criterion: any rank or SBC number. One realization cannot
# produce one, and reading a single-chain rank as calibration is the error
# this project has made before.
#
# Fresh output directory: pre-restoration chains and checkpoints are a
# different model. run_gibbs_chain now refuses a checkpoint whose
# packing_version or vector length does not match (PACKING_VERSION = 2), so a
# stale directory would abort rather than silently resume -- but do not point
# this at one anyway.
#
# Cost: ~24.8s/sweep measured at this configuration pre-restoration
# (job 11849969); the packed vector grows by lmax-2 = 62 coordinates out of
# ~2080, so expect ~5h for 700 sweeps. 24h walltime, checkpoint every 50.
#
# Job cap: check before submitting (limit is 200):
#   squeue -u dc-hick2 --noheader -r -o "%a" | grep -c durham
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

mkdir -p logs results/analysis/pilot_packing_v2_lmax64_nocl4
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "Packing-v2 validation, realization ${SLURM_ARRAY_TASK_ID}, lmax=${LMAX} (prior mass matrix, Block 4 OFF)"
echo "git HEAD: $(git rev-parse --short HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"
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
    --outdir results/analysis/pilot_packing_v2_lmax64_nocl4 \
    --checkpoint_every 50

rc=$?
echo "Realization ${SLURM_ARRAY_TASK_ID} exited with status $rc"
# Propagate the status so sacct shows FAILED rather than COMPLETED.
exit $rc
