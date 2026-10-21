#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=35-36
#SBATCH -J repro_cl_collapse
#SBATCH -o logs/repro_cl_collapse_%A_%a.out
#SBATCH -e logs/repro_cl_collapse_%A_%a.err
# ROADMAP T0.1 (2026-09-27): replay the MAP start + first sweeps of ν = 30 sky
# r036 (C_62 collapsed to ~1e-5 for the whole production chain) and r035
# (control), saving every sweep, to see whether the collapse is set by the MAP.
set -u
cd /cosma/apps/durham/dc-hick2/diffcmb
OUT=/cosma5/data/durham/dc-hick2/diffcmb/scratch_r036
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/diffcmb
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 DUCC0_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2 TF_ENABLE_ONEDNN_OPTS=0
.venv/bin/python -u scripts/coverage_ensemble_chain.py \
    --realization ${SLURM_ARRAY_TASK_ID} --lmax 64 --nside 64 \
    --n_burnin 0 --n_samples 300 --n_lfs 10 --phi_n_lfs 240 \
    --phi_mass_matrix prior --noisesig 30 --phi_amplitude 3000 \
    --lensing_operator exact --fiducial corrected \
    --outdir $OUT --checkpoint_every 50 --cl_phiphi_prior_nu 30.0
exit $?
