#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH -J bias_repl_aware
#SBATCH -o logs/bias_repl_aware_%A_%a.out
#SBATCH -e logs/bias_repl_aware_%A_%a.err
#SBATCH --exclusive
#SBATCH --array=1-3

# ===================================================================
# 2026-09-12 REPLICATE THE C_l^TT BIAS-REDUCTION RESULT ON 3 MORE SKIES.
#
# WHY. The 93% bias-reduction result (achievements.md, 2026-09-11) is the
# strongest result this project has -- lensing-aware consistent with unbiased
# in every reliable bin (|pull| <= 2.3) while the lensing-blind deficit grows
# monotonically to -5.4% at [100,128), a 172-sigma effect. It currently rests
# on ONE sky realization (seed=0). "Is this one realization?" is the first
# question a referee asks of a single-sim result, and the honest answer today
# is yes. The pulls are large enough that the effect is not statistical noise
# WITHIN the chain, but a single sky cannot exclude a lucky realization -- the
# lensing bias depends on the particular phi realization drawn.
#
# This takes the headline demonstration from N=1 to N=4. It is REPLICATION at
# a fixed configuration, not tuning: every sampler setting is identical to job
# 11966631 and only --seed changes (1, 2, 3 vs 0). It is therefore outside
# ROADMAP.md's "no further phi-equilibration tuning without sign-off" rule.
#
# Config IDENTICAL to job 11966631 (the seed=0 lensing-aware chain): lmax=128,
# nside=128, noisesig=1.0, HMC (NOT the script's mclmc default, which is a
# closed NO-GO), phi_mass_matrix='prior', phi_n_lfs=240, n_burnin=400,
# n_samples=1900, map_steps=2000, map_lr=0.01. Each array task draws its own
# sky+phi+noise from its seed.
#
# EXPECTED: each task will again print NO-GO on the equilibration gate, because
# phi does not equilibrate in [2,10) at lmax=128 (job 11966631, worst lag-1
# 0.967 -- re-confirmed post-fix). That is EXPECTED and does not invalidate the
# run: the bias-reduction figure is built from [10,30) upward, where every bin
# decorrelates within 10-50 lags. Do not read the NO-GO as a failed job.
#
# Cost: 25.7h/task at 48.6 s/sweep for 2300 sweeps, inside the 72h cap.
# checkpoint_every=50; resubmit against the same checkpoint if walltime is hit.
#
# Job cap: 3 array tasks (limit 200). Re-check before submitting:
#   squeue -u dc-hick2 --noheader -r -o "%a" | grep -c durham
#
# HARVEST. .err per task for tracebacks first (SLURM COMPLETED is not enough).
# Then run the matching blind baselines (submit_bias_reduction_replicate_blind
# .slurm, 4 min each) and compare per seed:
#   PYTHONPATH=diffcmb .venv/bin/python scripts/compare_cl_bias_reduction.py \
#       --blind results/analysis/lensing_blind_baseline_lmax128_seed<N>.npz \
#       --aware results/analysis/bias_repl_aware_lmax128_seed<N>.npz
# The claim to test is that the 0.0226 -> 0.0016 mean-|bias| gap reproduces in
# sign and rough size on all four skies, NOT that the numbers match exactly
# (each sky has its own phi realization, so the size of the blind deficit will
# vary). Report the across-seed spread, not just seed 0.
# ===================================================================

SEED=${SLURM_ARRAY_TASK_ID}
LMAX=128
NSIDE=128
NOISESIG=1.0
BURNIN=400
SAMPLES=1900
N_LFS=10
PHI_N_LFS=240
MAP_STEPS=2000
MAP_LR=0.01

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
mkdir -p logs results/analysis
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "=== bias-reduction replication, seed=${SEED} (job ${SLURM_JOB_ID}) ==="
echo "Running on: $(hostname) at $(date)"
echo "git HEAD: $(git rev-parse --short HEAD)"

$PYTHON -u scripts/pilot_coverage_equilibration.py \
    --lmax $LMAX \
    --nside $NSIDE \
    --noisesig $NOISESIG \
    --n_burnin $BURNIN \
    --map_steps $MAP_STEPS \
    --map_lr $MAP_LR \
    --n_samples $SAMPLES \
    --n_lfs $N_LFS \
    --phi_sampler hmc \
    --phi_mass_matrix prior \
    --phi_n_lfs $PHI_N_LFS \
    --seed $SEED \
    --checkpoint_path results/analysis/bias_repl_aware_lmax128_seed${SEED}_ckpt.npz \
    --checkpoint_every 50 \
    --out results/analysis/bias_repl_aware_lmax128_seed${SEED}.npz

echo "Exited with status $?"
