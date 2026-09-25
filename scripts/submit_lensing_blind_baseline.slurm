#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH -J lensing_blind_baseline
#SBATCH -o logs/lensing_blind_baseline_%j.out
#SBATCH -e logs/lensing_blind_baseline_%j.err
#SBATCH --exclusive

# Commander-style lensing-blind C_ℓ^TT baseline for the bias-reduction figure.
# ROADMAP.md §2.3: "C_ℓ^TT bias reduction vs a lensing-blind (Commander-style)
# analysis of the same sims."
#
# Runs Phase 0 unlensed Gibbs (alm + C_ℓ blocks only; NO phi block) on the
# IDENTICAL simulation as the MCLMC pilot (seed=0, lmax=128, nside=128,
# noisesig=1.0). Bit-for-bit same sky + noise = direct comparison is valid.
#
# This run is completely independent of the phi-block equilibration problem.
# We generate the reference lensing-blind posterior NOW so it is ready when
# the lensing-aware chain is equilibrated.
#
# Expected runtime: ~500 burnin + 3000 samples × ~3s/sweep (no phi block,
# much faster than the joint sampler) ≈ ~3h on dine2.
#
# Output: results/analysis/lensing_blind_baseline_lmax128.npz
#   cl_samples: (3000, lmax-2)  posterior C_ℓ^TT samples (unlensed model fit)
#   The bias vs the lensing-aware chain shows directly how much lensing
#   suppresses power in each ℓ bin when ignored.
LMAX=128
NSIDE=128
NOISESIG=1.0
N_BURNIN=500
N_SAMPLES=3000
SEED=0

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

echo "=== lensing_blind_baseline job ${SLURM_JOB_ID} ==="
echo "Running on: $(hostname)  at $(date)"
$PYTHON --version
free -h

$PYTHON -u scripts/run_lensing_blind_baseline.py \
    --lmax $LMAX \
    --nside $NSIDE \
    --noisesig $NOISESIG \
    --n_burnin $N_BURNIN \
    --n_samples $N_SAMPLES \
    --seed $SEED \
    --checkpoint_path results/analysis/lensing_blind_baseline_lmax128_ckpt.npz \
    --checkpoint_every 100 \
    --out results/analysis/lensing_blind_baseline_lmax128.npz

echo "=== Done at $(date) ==="
