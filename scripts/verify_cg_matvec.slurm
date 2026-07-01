#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH -J verify_cg_matvec
#SBATCH -o logs/verify_cg_matvec_%j.out
#SBATCH -e logs/verify_cg_matvec_%j.err

echo "===="
echo "Starting verify_cg_matvec $SLURM_JOB_ID at $(date)"
echo "Running on nodes: $SLURM_NODELIST"
echo "===="

cd /cosma/apps/durham/dc-hick2/diffcmb
mkdir -p logs

export PYTHONPATH=$(pwd)/diffcmb
export OMP_NUM_THREADS=4

.venv/bin/python scripts/verify_cg_matvec.py

echo "===="
echo "Finished at $(date)"
echo "===="
