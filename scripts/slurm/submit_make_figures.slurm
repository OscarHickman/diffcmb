#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma8-shm,cosma8-ska,cosma8-shm2,cosma8-shm3,cosma5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH -J make_figures
#SBATCH -o logs/make_figures_%j.out
#SBATCH -e logs/make_figures_%j.err
# `make figures` from the ensemble named by the Makefile's ENSEMBLE (D5: the
# nu = 30 long Block-4-ON ensemble). Figure 2 needs that directory's
# blind_rNNN.npz, so submit with --dependency=afterok:<blind array job>.
# Not dine2: results/analysis is a symlink into /cosma5, which dine2 does not mount.
set -u
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
mkdir -p logs
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 RAYON_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1
export TF_ENABLE_ONEDNN_OPTS=0
make figures
rc=$?
echo "Exited with status $rc"
exit $rc
