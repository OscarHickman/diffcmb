#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska,bluefield1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000G
#SBATCH --time=01:00:00
#SBATCH -J debug_cg
#SBATCH -o logs/debug_cg_%j.out
#SBATCH -e logs/debug_cg_%j.err
# Not dine2: results/analysis is a symlink into /cosma5, which dine2 does not mount
# (ROADMAP 2026-09-24). Memory sized from past MaxRSS instead of a whole node.

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
