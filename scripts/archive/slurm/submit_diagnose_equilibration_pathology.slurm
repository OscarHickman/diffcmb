#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -J diagnose_equilibration_pathology
#SBATCH -o logs/diagnose_equilibration_pathology_%j.out
#SBATCH -e logs/diagnose_equilibration_pathology_%j.err

# Diagnostic script for the lmax=128 MCLMC equilibration pathology.
# Post-processing only (numpy + matplotlib; no GPU, no TF forward model).
#
# Reads: results/analysis/pilot_coverage_lmax128_mclmc_n3300.npz
# Writes: results/analysis/diagnose_equilibration_pathology.png
#         (prints verdict table to stdout)
#
# Classifies the pathology as:
#   A) Step size too large (oscillation) — fine-lag ACF goes negative early then bounces
#   B) Burn-in incomplete (drift) — per-bin power level still trending
#   C) Long autocorrelation time (slow mode) — monotone ACF, no oscillation
#
# Then recommends the next compute experiment (step-size grid at nside=128, longer burn-in,
# or increased L) before re-running the full 3700-sweep equilibration pilot.

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

mkdir -p logs results/analysis
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=4

echo "=== diagnose_equilibration_pathology job ${SLURM_JOB_ID} ==="
echo "Running on: $(hostname)  at $(date)"
$PYTHON --version

$PYTHON -u scripts/archive/diagnose_equilibration_pathology.py \
    --npz results/analysis/pilot_coverage_lmax128_mclmc_n3300.npz \
    --lmax 128 \
    --out results/analysis/diagnose_equilibration_pathology.png

echo "=== Done at $(date) ==="
