#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH -J bench_lensing
#SBATCH -o logs/bench_lensing_%j.out
#SBATCH -e logs/bench_lensing_%j.err

echo "===="
echo "Starting bench_lensing $SLURM_JOB_ID at $(date)"
echo "Running on nodes: $SLURM_NODELIST"
echo "===="

cd /cosma/apps/durham/dc-hick2/diffcmb
mkdir -p logs

export PYTHONPATH=$(pwd)/diffcmb
export OMP_NUM_THREADS=4

.venv/bin/python scripts/benchmark_lensing.py

echo "===="
echo "Finished at $(date)"
echo "===="
