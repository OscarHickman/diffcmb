#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH -J debug_cg
#SBATCH -o logs/debug_cg_%j.out
#SBATCH -e logs/debug_cg_%j.err

echo "===="
echo "Starting debug_cg $SLURM_JOB_ID at $(date)"
echo "Running on nodes: $SLURM_NODELIST"
echo "===="

cd /cosma/apps/durham/dc-hick2/diffcmb
mkdir -p logs

export PYTHONPATH=$(pwd)/diffcmb
export OMP_NUM_THREADS=4

.venv/bin/python scripts/debug_cg.py \
    /cosma/apps/durham/dc-hick2/diffcmb/results/lmax300_nside256_gibbs_real_double

echo "===="
echo "Finished at $(date)"
echo "===="
