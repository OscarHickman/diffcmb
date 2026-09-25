#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J pilot_coverage_lmax128_extend
#SBATCH -o logs/pilot_coverage_lmax128_extend_%j.out
#SBATCH -e logs/pilot_coverage_lmax128_extend_%j.err
#SBATCH --exclusive

# Follow-up to job 11663477 (COMPLETED, verdict NO-GO by the script's fixed
# lag-1>=0.90 rule: worst lag-1 was 0.981 at 1000 sweeps). Local re-analysis of
# the saved traces (results/analysis/pilot_coverage_lmax128_v2.npz) extended
# the lag table past the script's own cutoff (lag<=50) out to lag=400 and found
# the autocorrelation decays smoothly through zero by lag~75-100 in all 5 l-bins
# (e.g. [30,60): 0.981 -> 0.927 -> 0.851 -> 0.690 -> 0.448 -> 0.223 -> -0.121 at
# lags 1/5/10/25/50/75/100) rather than staying pinned near 1 the way the
# pathological job 11612969 did (lag-1/lag-5 > 0.9999, no decay at all). That
# pattern -- real decay to noise, modest drift (worst 1.95 sigma, under the 2.0
# NO-GO) -- is the signature of a genuinely mixing, stationary chain with a
# longer-than-budgeted correlation time, not a stuck one. So this resubmission
# tests the "lengthen the window" remedy (ROADMAP.md's own second-preference
# option) rather than "drop lmax further" (the first-preference option, but one
# that would change the coverage test's target scale away from the intended
# lmax~100-150).
#
# This is a checkpoint RESUME, not a fresh run: run_gibbs_chain's resume path
# (diffcmb/samplers.py) restores the full chain state (alm/lncl/phi/C_L^phiphi/
# mass matrix/step sizes) from pilot_coverage_lmax128_v2_ckpt.npz and forces
# burnin_remaining=0 regardless of --n_burnin, so passing the same --n_burnin
# here is a no-op kept only for CLI clarity. Requesting --n_samples 2500 means
# n_samples_remaining = 2500-600 = 1900 additional sweeps on top of the 600
# already collected (1000 total already-paid-for sweeps carry over for free).
# At the measured ~32s/sweep (1000 sweeps / 8.91h from job 11663477), 1900
# sweeps is ~16.9h -- fits one 24h submission with margin.
#
# OOM risk: the confirmed ~362MB/sweep phi-block memory leak (achievements.md,
# "OPEN, unfixed") means cumulative sweeps (not lmax) sets the OOM ceiling,
# empirically ~1500-5500 sweeps depending on node free RAM. This job's final
# sweep count (2900) sits inside that range but not deep inside it -- if it
# OOMs, the checkpoint still preserves everything up to that point, so the
# worst case is a partial extension, not a lost run.
#
# --out intentionally differs from the v2 pilot's output file so the original
# 1000-sweep analysis (job 11663477) stays on disk for comparison against this
# extended one.
LMAX=128
NSIDE=128
BURNIN=400
SAMPLES=2500
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

echo "Extending coverage-test equilibration pilot at lmax=${LMAX} (nside=${NSIDE}) to ${SAMPLES} samples, phi_n_lfs=${PHI_N_LFS}"
echo "Resuming from results/analysis/pilot_coverage_lmax128_v2_ckpt.npz (600/600 already collected)"
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
    --out results/analysis/pilot_coverage_lmax128_v2_n2500.npz
