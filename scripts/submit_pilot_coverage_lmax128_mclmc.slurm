#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH -J pilot_coverage_lmax128_mclmc
#SBATCH -o logs/pilot_coverage_lmax128_mclmc_%j.out
#SBATCH -e logs/pilot_coverage_lmax128_mclmc_%j.err
#SBATCH --exclusive

# ROADMAP.md "MCLMC spike" GO branch (2026-08-08): gate job 11708844 showed
# MCLMC beating HMC ESS/wall-clock-second on all 6 probed l-bins at
# lmax=128/nside=64 (n_steps=30, step_size=0.1, L=200), including a 4x win at
# the previously-stuck l=[60,100) bin (HMC ESS/s=0.0010 vs MCLMC ESS/s=0.0040).
# This re-runs the actual lmax=128/nside=128 equilibration gate --
# pilot_coverage_equilibration.py's Block 3 default is now phi_sampler=mclmc
# -- to confirm the win transfers to the equilibration diagnostic (lag-1
# autocorrelation / drift), not just the small side-by-side ESS/s comparison.
#
# Fresh checkpoint, NOT a resume of any HMC checkpoint: the sampler itself
# changed, so an HMC-warmed phi state is not a valid MCLMC start. Same
# warm-started-away-from-truth protocol as every prior pilot job.
#
# n_samples=3300 matches the phi_n_lfs=240 HMC run's total window so the
# equilibration diagnostics are comparable sweep-for-sweep. Checkpointing
# every 50 sweeps (pilot_coverage_equilibration.py default) means a walltime
# timeout loses at most 50 sweeps of progress and the run resumes from
# --checkpoint_path automatically on requeue (run_gibbs_chain resumes
# alm/phi/C_L^phiphi state, not just the alm block).
#
# Walltime budget: 48h, double the 24h the phi_n_lfs=240 HMC run at this same
# lmax/nside/n_samples used (submit_pilot_coverage_lmax128_phi240.slurm) --
# generous headroom since the gate (job 11708844) showed MCLMC running
# *faster* wall-clock per comparable window than HMC at this dimensionality
# (3560s vs 8910s for the same 300-sweep window), so this is margin, not an
# expected requirement. dine2's partition MaxTime is 3-00:00:00 (72h).
LMAX=128
NSIDE=128
BURNIN=400
SAMPLES=3300
N_LFS=10
MAP_STEPS=2000
MAP_LR=0.01
PHI_SAMPLER=mclmc
PHI_HMC_STEP_SIZE=0.1
PHI_N_LFS=30
PHI_MCLMC_L=200.0

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

echo "MCLMC-production lmax=128 pilot re-run: phi_sampler=${PHI_SAMPLER}, n_steps=${PHI_N_LFS}, step_size=${PHI_HMC_STEP_SIZE}, L=${PHI_MCLMC_L}"
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
    --phi_mclmc_L $PHI_MCLMC_L \
    --checkpoint_path results/analysis/pilot_coverage_lmax128_mclmc_ckpt.npz \
    --checkpoint_every 50 \
    --out results/analysis/pilot_coverage_lmax128_mclmc_n3300.npz
