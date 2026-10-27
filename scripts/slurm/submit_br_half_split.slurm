#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska,cosma8-ska2,cosma7-shm,bluefield1
#SBATCH --exclude=mad01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -J br_half
#SBATCH -o logs/br_half_split_%j.out
#SBATCH -e logs/br_half_split_%j.err
# ROADMAP T1.4: is the flat-prior calibration's z spread (1.18-1.26 without r036, job 12201039)
# the Blackwell-Rao Monte Carlo noise? Half-chain posteriors on both calibration ensembles.
set -u
cd /cosma/apps/durham/dc-hick2/diffcmb
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/diffcmb:$(pwd)/scripts
NT=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT DUCC0_NUM_THREADS=$NT
export RAYON_NUM_THREADS=$NT NUMBA_NUM_THREADS=$NT TF_NUM_INTRAOP_THREADS=$NT TF_NUM_INTEROP_THREADS=1
A=results/analysis
echo "== flat-prior full sky, lcdm"
.venv/bin/python -u scripts/diagnostics/br_half_split.py $A/ens_exact_l64_A3000_n30_nu30_long --target lcdm || exit 1
echo "== strict real sky, hier"
.venv/bin/python -u scripts/diagnostics/br_half_split.py $A/ens_realsky_l64_A3000_n30_nu30_strict --target hier || exit 1
