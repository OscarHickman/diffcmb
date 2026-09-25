#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J pilot_coverage_lmax64_hmc_extended
#SBATCH -o logs/pilot_coverage_lmax64_hmc_extended_%j.out
#SBATCH -e logs/pilot_coverage_lmax64_hmc_extended_%j.err
#SBATCH --exclusive

# ROADMAP.md trigger-3 option 2/3, launched 2026-08-16 alongside the lmax=32
# and phi_n_lfs-raise jobs (user decision: "do all of them").
#
# Job 11752452 (lmax=64, n_burnin=400+n_samples=3300 = 3700 sweeps) got 3/4
# l-bins past the equilibration gate; only the top bin [60,64) failed, and
# its ACF was STILL DECAYING at lag 200 (r_150=0.431 -> r_200=0.342), unlike
# the lmax=128 pilots' bins which plateaued or reversed. This is the
# "lengthen the window" option: resume that exact chain (same seed, same
# config) from a COPY of its final checkpoint
# (pilot_coverage_lmax64_hmc_extended_ckpt.npz, copied 2026-08-16 from the
# completed run's archived checkpoint so the original n3300 artifact stays
# untouched and reproducible) and run to double the total collected samples.
#
# n_samples=6600 (vs the original 3300): run_gibbs_chain's checkpoint resume
# (samplers.py) skips burn-in and continues collecting toward n_samples, so
# this adds another ~3300 samples on top of the 3300 already banked, not a
# restart. Same phi_n_lfs=240/phi_hmc_step_size=0.05 throughout -- only the
# window length changes, isolating this lever from the phi_n_lfs-raise job.
LMAX=64
NSIDE=64
BURNIN=400
SAMPLES=6600
N_LFS=10
PHI_SAMPLER=hmc
PHI_HMC_STEP_SIZE=0.05
PHI_N_LFS=240
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
ARCHIVE_DIR="/cosma8/data/dp004/dc-hick2/diffcmb_results_archive"
CKPT_TARGET="${ARCHIVE_DIR}/pilot_coverage_lmax64_hmc_extended_ckpt.npz"
CKPT_LINK="results/analysis/pilot_coverage_lmax64_hmc_extended_ckpt.npz"
if [ ! -e "$CKPT_LINK" ]; then
    ln -s "$CKPT_TARGET" "$CKPT_LINK"
fi

export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "HMC production-path fallback, WINDOW EXTENSION: lmax=${LMAX} (nside=${NSIDE}), target n_samples=${SAMPLES} (resumed from a 3300-sample checkpoint copy), phi_n_lfs=${PHI_N_LFS}"
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
    --phi_sampler $PHI_SAMPLER \
    --phi_hmc_step_size $PHI_HMC_STEP_SIZE \
    --phi_n_lfs $PHI_N_LFS \
    --checkpoint_path "$CKPT_LINK" \
    --checkpoint_every 50 \
    --out results/analysis/pilot_coverage_lmax64_hmc_extended_n6600.npz
