#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J pilot_coverage_lmax128_phi240
#SBATCH -o logs/pilot_coverage_lmax128_phi240_%j.out
#SBATCH -e logs/pilot_coverage_lmax128_phi240_%j.err
#SBATCH --exclusive

# ROADMAP.md "Sampler-lever decision" step 2: the bounded phi_n_lfs raise,
# triggered by two pieces of 2026-08-06 evidence: (a) the extended lmax=128
# pilot's l=[60,100) bin got WORSE with more window (lag-1 0.981 -> 0.996,
# still >0.2 at lag 200 -- see reanalyze_pilot_checkpoint.py's re-analysis of
# job 11665429's checkpoint), so "lengthen the window" is exhausted for that
# bin; (b) the free phi-deficit-vs-S/N diagnostic (analyze_phi_deficit_vs_snr.py)
# found the lmax=300 deficit worsens at LOW S/N (rho=-0.78), the
# under-mixing/Wiener-suppressed-start signature, not sampler geometry -- so
# per the decision rule this is a "try the cheap lever" case, not an MCLMC
# trigger.
#
# Budget (this run's bounded choice, absent a different explicit number):
# phi_n_lfs 80 -> 240 (3x, middle of the roadmap's 2-4x range). Resumes from
# a COPY of the 2500-sample v2 checkpoint (pilot_coverage_lmax128_v3_phi240_ckpt.npz,
# copied 2026-08-06 so the original phi_n_lfs=80 checkpoint stays on disk for
# comparison, same discipline as submit_pilot_coverage_lmax128_extend.slurm).
# A fresh SLURM job/process also resets the known phi-block memory-leak
# accumulator (achievements.md), so resuming here does not inherit the
# previous job's leak exposure.
#
# n_samples=3300 means 800 additional sweeps on top of the 2500 already
# collected. Cost estimate: 80->240 leapfrog steps is at most a 3x per-sweep
# cost increase if the phi block dominates wall-clock (conservative upper
# bound); at the v2 pilot's measured ~32s/sweep (phi_n_lfs=80, lmax=128),
# that is <=~96s/sweep, so 800 sweeps is <=~21.3h -- fits one 24h submission
# with margin. If it runs closer to the alm-block-dominated lower bound, it
# finishes with more margin, not less.
#
# Judgment call: whether tripling phi_n_lfs measurably shortens the l=[60,100)
# bin's autocorrelation is the open question this run answers, not assumed.
LMAX=128
NSIDE=128
BURNIN=400
SAMPLES=3300
N_LFS=10
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
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "Bounded phi_n_lfs raise: lmax=${LMAX} (nside=${NSIDE}), phi_n_lfs=${PHI_N_LFS} (was 80), to ${SAMPLES} samples"
echo "Resuming from results/analysis/pilot_coverage_lmax128_v3_phi240_ckpt.npz (2500/2500 already collected at phi_n_lfs=80)"
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
    --checkpoint_path results/analysis/pilot_coverage_lmax128_v3_phi240_ckpt.npz \
    --checkpoint_every 50 \
    --out results/analysis/pilot_coverage_lmax128_v3_phi240_n3300.npz
