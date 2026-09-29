#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -J alm_band_edge
#SBATCH -o logs/alm_band_edge_%j.out
#SBATCH -e logs/alm_band_edge_%j.err
# ROADMAP T0.1 (2026-09-27): a_lm band-edge breakdown (per l, per m, thinning,
# quarters, per-mode z and shrinkage) on every exact-operator ensemble.
# Optional: INDIRS (space-separated), default every lmax=64 exact ensemble.
set -u
cd /cosma/apps/durham/dc-hick2/diffcmb
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/diffcmb
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 DUCC0_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1
A=results/analysis
INDIRS=${INDIRS:-"$A/ens_exact_l64_A3000_n30_nu30_long $A/ens_exact_l64_A3000_n30_nocl4 $A/ens_exact_l64_A3000_n30_long $A/ens_exact_l64_A3000_n30"}
ARGS=""
for d in $INDIRS; do ARGS="$ARGS --indir $d"; done
.venv/bin/python -u scripts/diagnose_alm_band_edge.py $ARGS
exit $?
