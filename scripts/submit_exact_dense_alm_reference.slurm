#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-47
#SBATCH -J dense_alm_ref
#SBATCH -o logs/dense_alm_ref_%A_%a.out
#SBATCH -e logs/dense_alm_ref_%A_%a.err
# ROADMAP T0.1 (2026-09-27): exact dense reference for the a_lm power rank,
# phi fixed at each sky's truth; one sky per task. Optional ENSEMBLE (source
# chains) and OUT.
set -u
cd /cosma/apps/durham/dc-hick2/diffcmb
ENSEMBLE=${ENSEMBLE:-results/analysis/ens_exact_l64_A3000_n30_nu30_long}
OUT=${OUT:-results/analysis/exact_dense_alm_reference_l64_nu30}
R=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/diffcmb:$(pwd)/scripts
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 DUCC0_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2 TF_ENABLE_ONEDNN_OPTS=0
.venv/bin/python -u scripts/exact_dense_alm_reference.py --chain $ENSEMBLE/chain_r$R.npz --outdir $OUT
exit $?
