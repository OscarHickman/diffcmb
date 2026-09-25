#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J pilot_coverage_lmax64_hmc_ordfix
#SBATCH -o logs/pilot_coverage_lmax64_hmc_ordfix_%j.out
#SBATCH -e logs/pilot_coverage_lmax64_hmc_ordfix_%j.err
#SBATCH --exclusive

# Re-run of submit_pilot_coverage_lmax64_hmc.slurm (job 11752452, NO-GO,
# worst lag-1 0.975 in bin [60,64)) after fixing lensing.py's alm ordering
# bug (commit a16e9e6, 2026-08-24): _alm_packed_to_hp/_alm_hp_to_packed
# never called almmotho/almhotmo, so phi_packed was handed to healpy/ducc in
# author (L-major) ordering while they read m-major -- every coefficient sat
# at the wrong multipole, so the per-L C_L^phiphi prior (Block 3's Gaussian
# term) acted on modes the likelihood's SHT placed at a different L. This is
# the leading candidate explanation for the whole phi-equilibration
# pathology: a mismatched prior/likelihood diagonal produces exactly the
# un-preconditionable geometry the block-Nystrom/Fisher machinery was built
# to chase. Run the SIMPLEST configuration first -- default
# phi_mass_matrix='prior' (diagonal), no Nystrom correction -- before
# assuming any of that machinery is still needed post-fix.
#
# Identical config to job 11752452 otherwise (lmax=64, nside=64,
# phi_n_lfs=240, phi_hmc_step_size=0.05, Block 4 ON via
# pilot_coverage_equilibration.py's default) so the comparison is
# apples-to-apples: same everything except the ordering bug.
#
# Fresh checkpoint (separate archive path from the pre-fix run, which must
# not be resumed from -- its state was built on scrambled phi).
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
CKPT_TARGET="${ARCHIVE_DIR}/pilot_coverage_lmax64_hmc_ordfix_ckpt.npz"
CKPT_LINK="results/analysis/pilot_coverage_lmax64_hmc_ordfix_ckpt.npz"
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

echo "Post-ordering-fix HMC baseline: lmax=${LMAX} (nside=${NSIDE}), phi_n_lfs=${PHI_N_LFS}, phi_hmc_step_size=${PHI_HMC_STEP_SIZE}"
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
    --out results/analysis/pilot_coverage_lmax64_hmc_ordfix_n3300.npz
