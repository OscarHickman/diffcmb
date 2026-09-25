#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J validate_sim_lmax300_phi80
#SBATCH -o logs/validate_sim_lmax300_phi80_%j.out
#SBATCH -e logs/validate_sim_lmax300_phi80_%j.err
#SBATCH --exclusive

# ROADMAP.md Section 1, "Simulation validation" re-submission with the
# leapfrog-schedule fix (achievements.md, 2026-07-22): the original run
# (job 11612969, phi_n_lfs=10) failed its point-agreement check because the
# phi block barely moved (lag-1/lag-5 autocorrelation >0.9999). A 5-seed
# local smoke test (scripts/smoke_phi_leapfrog_schedule.py) found that
# lengthening the phi HMC trajectory (not reconditioning the mass matrix --
# the Fisher-mass-matrix route was tried and closed as unfavorable) fixes
# this: mean autocorrelation fell monotonically 0.745->0.576->0.541 for
# phi_n_lfs 20->80->200, with most of the gain already at 80. A local
# lmax=300 cost check (scripts/smoke_lmax300_matrixfree_lensing.py) measured
# phi_n_lfs=100 at 229.67s/sweep vs phi_n_lfs=10's 44.03s/sweep (5.2x); 80 is
# expected close to that same scaling (~185s/sweep). At 1200 sweeps
# (200 burn-in + 1000 samples) that's ~62 CPU-hours total, so this run is
# expected to need 3-4 sequential 24h submissions -- resubmit this same
# script after each walltime timeout; it resumes from its own checkpoint.
#
# Uses fresh checkpoint/output paths (not the failed run's
# validate_sim_lmax300_ckpt.npz / validate_sim_lmax300.npz) since that
# checkpoint already holds a complete 1000-sample run at the old phi_n_lfs=10
# settings and must not be resumed into under the new schedule.
LMAX=300
NSIDE=256
BURNIN=200
SAMPLES=1000
N_LFS=10
PHI_N_LFS=80

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

echo "Starting lmax=300 simulation validation run, phi_n_lfs=${PHI_N_LFS} (nside=${NSIDE})"
echo "Using Python: $($PYTHON --version 2>&1)"
free -h

$PYTHON -u scripts/validate_sim_lmax300_lensing.py \
    --lmax $LMAX \
    --nside $NSIDE \
    --n_burnin $BURNIN \
    --n_samples $SAMPLES \
    --n_lfs $N_LFS \
    --phi_n_lfs $PHI_N_LFS \
    --checkpoint_path results/analysis/validate_sim_lmax300_phi80_ckpt.npz \
    --checkpoint_every 50 \
    --out results/analysis/validate_sim_lmax300_phi80.npz
