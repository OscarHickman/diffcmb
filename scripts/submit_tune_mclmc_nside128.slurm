#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J tune_mclmc_nside128
#SBATCH -o logs/tune_mclmc_nside128_%j.out
#SBATCH -e logs/tune_mclmc_nside128_%j.err
#SBATCH --exclusive

# MCLMC step_size/L tuning grid at the *production* nside=128.
#
# MOTIVATION (ROADMAP.md, achievements.md):
#   Job 11710475 (MCLMC n_steps=30, step_size=0.1, L=200, nside=128) returned
#   NO-GO. The [2,10) bin ACF is non-monotonic: goes to -0.084 at lag 50, bounces
#   back to +0.499 at lag 100 -- classic overshoot/oscillation, not slow decay.
#   step_size=0.1 was tuned at nside=64 (job 11708844); the whitened-space curvature
#   at nside=128 differs, making step_size=0.1 too large. This job finds the
#   non-oscillating step_size before relaunching the full 3700-sweep pilot.
#
# GRID: step_sizes={0.005,0.01,0.02,0.05} x L={100,200}
#   = 8 configs x ~(150 burnin + 200 samples) sweeps at ~18.5s/sweep = ~10.3h total
#   (conservative; likely faster at smaller step_size since fewer grad evals diverge)
#
# OUTPUT: logs/tune_mclmc_nside128_<jobid>.out (per-config ACF table + summary)
#         results/analysis/mclmc_tune_nside128/*.npz (phi_samples per config)
#
# NEXT STEP: read the summary table at the end of the log. Pick the smallest
#   step_size with no oscillation in [2,10), then re-run
#   submit_pilot_coverage_lmax128_mclmc.slurm with that step_size/L.
LMAX=128
NSIDE=128
N_STEPS=30
STEP_SIZES="0.005,0.01,0.02,0.05"
L_VALS="100,200"
N_BURNIN=150
N_SAMPLES=200

# results/analysis is a symlink into /cosma5, which dine2 does not mount (ROADMAP
# 2026-09-24). Fail loudly here rather than at the first write.
if [ ! -d /cosma/apps/durham/dc-hick2/diffcmb/results/analysis/ ]; then
    echo "results/analysis (/cosma5) is not mounted on $(hostname); relocate outputs before running on dine2" >&2
    exit 1
fi
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

mkdir -p logs results/analysis/mclmc_tune_nside128
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "=== tune_mclmc_nside128 job ${SLURM_JOB_ID} ==="
echo "Running on: $(hostname)  at $(date)"
$PYTHON --version
free -h

$PYTHON -u scripts/tune_mclmc_step_size_nside128.py \
    --lmax $LMAX \
    --nside $NSIDE \
    --n_steps $N_STEPS \
    --step_sizes "$STEP_SIZES" \
    --L_vals "$L_VALS" \
    --n_burnin $N_BURNIN \
    --n_samples $N_SAMPLES \
    --out_dir results/analysis/mclmc_tune_nside128

echo "=== Done at $(date) ==="
