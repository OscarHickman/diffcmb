#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J pilot_coverage_lmax64_hmc
#SBATCH -o logs/pilot_coverage_lmax64_hmc_%j.out
#SBATCH -e logs/pilot_coverage_lmax64_hmc_%j.err
#SBATCH --exclusive

# ROADMAP.md "Sampler-lever decision", trigger 3(a)/(b) reached 2026-08-14
# (job 11744132: MCLMC NO-GO at lmax=128 even with the tuned mass matrix).
# User decision 2026-08-14: pursue a geometry fix in parallel
# (submit_diagnose_mclmc_fisher_geometry.slurm) AND this lower-scale
# fallback -- the pilot script's own NO-GO message's first-preference option
# ("drop lmax further"), never previously tried.
#
# Deliberately uses the VALIDATED PRODUCTION HMC path (phi_sampler='hmc',
# unchanged from before the MCLMC spike), not MCLMC -- this decouples the
# fallback from the still-open MCLMC geometry question entirely. If this
# clears the equilibration gate, the coverage ensemble (ROADMAP.md Section 1
# Priority 1) can proceed on HMC at lmax=64 regardless of what the Fisher
# spike finds, unblocking the critical path either way.
#
# lmax=64, nside=64 (half of the failed lmax=128/nside=128 configuration,
# still comfortably above the lmax<=50 scale where achievements.md's Phase 2
# pre-production gate found (alm, phi) mixing "marginal, not pathological").
# phi_n_lfs=240, phi_hmc_step_size=0.05: the best previously-recorded HMC
# configuration (job 11694912, lmax=128/nside=128) -- reused as-is rather
# than re-tuned, since this run's job is to test whether scale alone fixes
# the pathology; a step-size retune at nside=64 is a separate lever if this
# comes back ambiguous rather than a clean GO/NO-GO.
#
# Block 4 (C_L^phiphi|phi) ON, phi_mass_matrix='prior' (forced) -- this is
# scripts/pilot_coverage_equilibration.py's default, unmodified production
# configuration (unlike the Fisher spike's dedicated script), so a GO here
# is directly usable for the coverage ensemble with no further reconciliation
# needed.
#
# Fresh checkpoint. Checkpoint on /cosma8 archive from the start per the
# 2026-08-13 lesson (achievements.md): /cosma/apps is NFS-quota-shared
# across all dc-hick2 projects; large checkpoints belong on
# /cosma8/data/dp004/dc-hick2/diffcmb_results_archive/, confirmed mounted on
# dine2 via srun.
#
# n_samples=3300 matches every prior pilot's window. lmax=64 is
# substantially cheaper per-sweep than lmax=128 (SHT cost scales steeply
# with lmax), so 24h is a comfortable budget, not a tight one.
LMAX=64
NSIDE=64
BURNIN=400
SAMPLES=3300
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
mkdir -p "$ARCHIVE_DIR"
CKPT_TARGET="${ARCHIVE_DIR}/pilot_coverage_lmax64_hmc_ckpt.npz"
CKPT_LINK="results/analysis/pilot_coverage_lmax64_hmc_ckpt.npz"
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

echo "HMC production-path fallback: lmax=${LMAX} (nside=${NSIDE}), phi_n_lfs=${PHI_N_LFS}, phi_hmc_step_size=${PHI_HMC_STEP_SIZE}"
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
    --out results/analysis/pilot_coverage_lmax64_hmc_n3300.npz
