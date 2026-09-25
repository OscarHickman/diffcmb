#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J pilot_coverage_lmax128
#SBATCH -o logs/pilot_coverage_lmax128_%j.out
#SBATCH -e logs/pilot_coverage_lmax128_%j.err
#SBATCH --exclusive

# ROADMAP.md Section 1, Priority 1: the prerequisite equilibration gate for the
# multi-realization rank/coverage test. NOT the coverage test itself -- this is
# the single pilot chain that decides whether the O(10-20)-chain ensemble is
# worth launching, per the 2026-07-29 scope re-set.
#
# Runs concurrently with (and independently of) the lmax=300 validation run
# 11661265 / 11663099 by design: the scope re-set explicitly scoped this below
# lmax=300 rather than waiting on that run's outcome, because a converged
# coverage test at lmax=128 is worth more than a non-converged point check at
# lmax=300.
#
# Two things make this different from submit_validate_sim_lmax300_phi80.slurm,
# and both are the point:
#   * warm-started AWAY from truth (independent prior draws for alm and phi),
#     since a rank statistic from a truth-warm-started chain inherits its
#     initialisation;
#   * Block 4 on (sample_cl_phiphi=True), required for the C_L^phiphi rank
#     statistic to exist -- which forces phi_mass_matrix='prior' (mutually
#     exclusive with 'fisher' by design; fisher is closed as unfavorable).
#
# Cost expectation: lmax=300/phi_n_lfs=80 measured 115-190s/sweep (node
# dependent). lmax=128 at nside=128 is ~1 order of magnitude cheaper, so 700
# sweeps should fit inside one 24h walltime with room to spare. Checkpointed
# anyway -- resubmit this same script if it times out; it resumes.
#
# v2 (2026-07-30), after job 11663105 froze. That run used a cold prior-draw alm
# start with 100 burn-in sweeps and no MAP pre-solve; the phi block locked at
# accept 0.004 with its low-L power ~240x the fiducial. v2 adds the data-driven
# MAP alm start (map_steps=2000, map_lr=0.01 -- the validated production settings
# from achievements.md, NOT find_map_estimate's much weaker function defaults)
# and raises burn-in to 400. Fresh checkpoint and
# output paths (*_v2_*) so the frozen state can never be resumed into.
#
# phi_n_lfs=80 is carried over deliberately: it is the validated setting from
# the leapfrog-schedule fix (achievements.md), and phi_n_lfs is the parameter
# that actually controls phi mixing. If the gate returns NO-GO, raising it is
# the third option, after dropping lmax and lengthening the window.
LMAX=128
NSIDE=128
BURNIN=400
SAMPLES=600
N_LFS=10
PHI_N_LFS=80
MAP_STEPS=2000
MAP_LR=0.01

# results/analysis is a symlink into /cosma5, which dine2 does not mount (ROADMAP
# 2026-09-24). Fail loudly here rather than at the first write.
if [ ! -d /cosma/apps/durham/dc-hick2/diffcmb/results/analysis/ ]; then
    echo "results/analysis (/cosma5) is not mounted on $(hostname); relocate outputs before running on dine2" >&2
    exit 1
fi
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

mkdir -p logs results/analysis
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "Starting coverage-test equilibration pilot at lmax=${LMAX} (nside=${NSIDE}), phi_n_lfs=${PHI_N_LFS}"
echo "Using Python: $($PYTHON --version 2>&1)"
free -h

$PYTHON -u scripts/pilot_coverage_equilibration.py \
    --lmax $LMAX \
    --nside $NSIDE \
    --n_burnin $BURNIN \
    --map_steps $MAP_STEPS \
    --map_lr $MAP_LR \
    --n_samples $SAMPLES \
    --n_lfs $N_LFS \
    --phi_n_lfs $PHI_N_LFS \
    --checkpoint_path results/analysis/pilot_coverage_lmax128_v2_ckpt.npz \
    --checkpoint_every 50 \
    --out results/analysis/pilot_coverage_lmax128_v2.npz
