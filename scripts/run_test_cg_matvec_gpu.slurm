#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH -J test_cg_matvec_gpu
#SBATCH -o logs/test_cg_matvec_gpu_%j.out
#SBATCH -e logs/test_cg_matvec_gpu_%j.err

echo "===="
echo "Starting test_cg_matvec_gpu $SLURM_JOB_ID at $(date)"
echo "Running on nodes: $SLURM_NODELIST"
echo "===="

cd /cosma/apps/durham/dc-hick2/diffcmb
mkdir -p logs

export PYTHONPATH=$(pwd)/diffcmb

.venv/bin/python -m pytest tests/test_cg_matvec.py -v -s

echo "===="
echo "Finished at $(date)"
echo "===="
