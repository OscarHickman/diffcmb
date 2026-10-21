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
export OMP_NUM_THREADS=1

echo "=== Phase 1 Lensing Tests ==="
echo "Python: $(.venv/bin/python --version 2>&1)"
echo "Node: $(hostname)"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "ldd healpy extension:"
ldd .venv/lib64/python3.9/site-packages/healpy/_sphtools.cpython-39-x86_64-linux-gnu.so 2>&1 | grep -i healpix
date

.venv/bin/python -c "
import faulthandler, sys
faulthandler.enable(file=sys.stderr)
import pytest
sys.exit(pytest.main(['-v', '--tb=short', 'tests/test_lensing.py']))
" 2>&1
EXIT_CODE=$?

echo "---"
date
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
