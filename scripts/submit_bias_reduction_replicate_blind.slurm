#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J bias_repl_blind
#SBATCH -o logs/bias_repl_blind_%A_%a.out
#SBATCH -e logs/bias_repl_blind_%A_%a.err
#SBATCH --array=1-3

# ===================================================================
# 2026-09-12 Lensing-blind baselines matching the 3 replication skies of
# submit_bias_reduction_replicate_aware.slurm (seeds 1, 2, 3).
#
# Each must be the SAME sky+phi+noise as the matching lensing-aware task, which
# is guaranteed by passing the same --seed: run_lensing_blind_baseline.py
# replicates pilot_coverage_equilibration.py's RNG setup exactly (rng_truth from
# seed, rng_noise from seed+20000), lenses the truth sky through phi_true and
# adds noise, then fits an UNLENSED model to it. compare_cl_bias_reduction.py
# additionally asserts the two chains' alm_true_packed agree before comparing,
# so a seed mismatch fails loudly rather than producing a silent nonsense figure.
#
# Runs on cosma5, not dine2: no phi block means 0.1 s/sweep (~4 min for 3500
# sweeps -- job 11980570), and dine2 is saturated by the aware replication and
# the Block-4-ON ensemble extension. Both inputs/outputs are real files on NFS,
# so no /cosma5 mount is needed.
#
# Config identical to job 11980570 apart from --seed.
# ===================================================================

SEED=${SLURM_ARRAY_TASK_ID}
LMAX=128
NSIDE=128
NOISESIG=1.0
N_BURNIN=500
N_SAMPLES=3000

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
mkdir -p logs results/analysis
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0

echo "=== lensing-blind baseline, seed=${SEED} (job ${SLURM_JOB_ID}) ==="

$PYTHON -u scripts/run_lensing_blind_baseline.py \
    --lmax $LMAX \
    --nside $NSIDE \
    --noisesig $NOISESIG \
    --n_burnin $N_BURNIN \
    --n_samples $N_SAMPLES \
    --seed $SEED \
    --checkpoint_path results/analysis/lensing_blind_baseline_lmax128_seed${SEED}_ckpt.npz \
    --checkpoint_every 100 \
    --out results/analysis/lensing_blind_baseline_lmax128_seed${SEED}.npz

echo "Exited with status $?"
