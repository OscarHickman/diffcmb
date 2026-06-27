#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -J lensing_tests
#SBATCH -o logs/lensing_tests_%j.out
#SBATCH -e logs/lensing_tests_%j.err

# Phase 1: Run gradient validation tests for the differentiable lensing operator.
# Tests cover: deflection, alm round-trip, bilinear weights, dL/dT, dL/dphi_alm.

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

mkdir -p logs

export PYTHONPATH=$(pwd)/diffcmb
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4

echo "=== Phase 1 Lensing Tests ==="
echo "Python: $(.venv/bin/python --version 2>&1)"
echo "Node: $(hostname)"
date

.venv/bin/python -m pytest tests/test_lensing.py -v --tb=short 2>&1
EXIT_CODE=$?

echo "---"
date
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
