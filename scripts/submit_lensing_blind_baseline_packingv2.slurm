#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH -J lensing_blind_baseline_packingv2
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
# Output: results/analysis/lensing_blind_baseline_lmax128_packingv2.npz
#   cl_samples: (3000, lmax-2)  posterior C_ℓ^TT samples (unlensed model fit)
#   The bias vs the lensing-aware chain shows directly how much lensing
#   suppresses power in each ℓ bin when ignored.
# ===================================================================
# 2026-09-11 RE-RUN UNDER THE RESTORED 2L+1 PACKING.
#
# WHY. The existing results/analysis/lensing_blind_baseline_lmax128.npz
# (written 2026-08-12) is packed at the OLD 2L width -- its alm_true_packed
# is 16254 long where packed_length(128) is now 16380 (a 126 = lmax-2
# shortfall, exactly the missing Im(a_{L,1}) dof). It therefore predates
# BOTH the 2026-08-24 alm-ordering fix and the 2026-09-06 dof restoration,
# and is a different model from the lensing-aware chain it is supposed to
# be the reference for. ROADMAP.md lists the lensing-blind side as "done";
# it is not, under the current packing.
#
# Config is byte-identical to the original submit script (seed=0, lmax=128,
# nside=128, noisesig=1.0), which is the same simulation the lensing-aware
# pilot job 11966631 drew -- so re-running it here regenerates the truth sky
# WITH the restored Im(a_{L,1}), matching that chain and making the C_l^TT
# bias-reduction comparison valid again. Truth/data are not bit-identical to
# the 2026-08-12 file at the same seed, which is the point.
#
# Output goes to a NEW path (..._packingv2.npz); the stale file is left in
# place as the pre-restoration reference, not overwritten.
#
# HARVEST. Check .err for a traceback, confirm alm_true_packed is 16380
# long, then compare cl_samples' mean bin-by-bin against the lensing-aware
# chain results/analysis/pilot_coverage_lmax128_postfix_hmc.npz.
# ===================================================================

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
    --checkpoint_path results/analysis/lensing_blind_baseline_lmax128_packingv2_ckpt.npz \
    --checkpoint_every 100 \
    --out results/analysis/lensing_blind_baseline_lmax128_packingv2.npz

echo "=== Done at $(date) ==="
