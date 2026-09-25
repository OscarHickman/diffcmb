#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J pilot_coverage_lmax64_hmc_philfs480
#SBATCH -o logs/pilot_coverage_lmax64_hmc_philfs480_%j.out
#SBATCH -e logs/pilot_coverage_lmax64_hmc_philfs480_%j.err
#SBATCH --exclusive

# ROADMAP.md trigger-3 option 3/3, launched 2026-08-16 alongside the lmax=32
# and window-extension jobs (user decision: "do all of them").
#
# Job 11752452 (lmax=64, phi_n_lfs=240) got 3/4 l-bins past the equilibration
# gate; only the top bin [60,64) failed (lag-1=0.975, drift=-1.99sigma).
# phi_n_lfs is the parameter with the best track record of controlling phi
# mixing in this project (achievements.md's leapfrog-schedule entry) --
# doubling it here (240->480) targets the one remaining failing bin directly,
# cheap to try given lmax=64's ~11.3s/sweep cost (job 11752452).
#
# Fresh checkpoint (not resumed from the phi_n_lfs=240 run's chain -- HMC
# trajectory length is a sampler-tuning parameter, not safe to change
# mid-chain without re-adapting the leapfrog step size from scratch).
# Same seed/data/MAP-start procedure as job 11752452 otherwise, so this
# isolates phi_n_lfs as the only changed variable relative to that run.
LMAX=64
NSIDE=64
BURNIN=400
SAMPLES=3300
N_LFS=10
PHI_SAMPLER=hmc
PHI_HMC_STEP_SIZE=0.05
PHI_N_LFS=480
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
mkdir -p "$ARCHIVE_DIR"
CKPT_TARGET="${ARCHIVE_DIR}/pilot_coverage_lmax64_hmc_philfs480_ckpt.npz"
CKPT_LINK="results/analysis/pilot_coverage_lmax64_hmc_philfs480_ckpt.npz"
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

echo "HMC production-path fallback, RAISED phi_n_lfs: lmax=${LMAX} (nside=${NSIDE}), phi_n_lfs=${PHI_N_LFS} (2x job 11752452's 240), phi_hmc_step_size=${PHI_HMC_STEP_SIZE}"
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
    --out results/analysis/pilot_coverage_lmax64_hmc_philfs480_n3300.npz
