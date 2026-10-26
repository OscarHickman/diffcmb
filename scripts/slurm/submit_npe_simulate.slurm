#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=6:00:00
#SBATCH --array=0-4
#SBATCH -J npe_sim
#SBATCH -o logs/npe_sim_%A_%a.out
#SBATCH -e logs/npe_sim_%A_%a.err
# ROADMAP T1.1 (2026-10-07): training and held-out sets for the spectra-level NPE,
# from scripts/npe/npe_simulate.py (the strict-mode generative process of the exact
# strict ensemble, nu = 30 on both spectra, A_phi = 3000, sigma = 30, lmax = nside = 64).
#   tasks 0-3: training, 50 000 sims each, realizations 100000-299999
#   task  4  : held-out SBC set, 10 000 sims, realizations 500000-509999
# The exact ensemble's skies are 0-47, so none of them is in either set.
# Not dine2: results/analysis is a symlink into /cosma5, which dine2 does not mount.
set -u
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
# SUMMARY_SET=v2 (T1.1 width gap, 2026-10-14): richer summaries plus the data a_lm,
# written to a separate directory; the default reproduces the v1 set.
SUMMARY_SET=${SUMMARY_SET:-v1}
OUT=results/analysis/npe_sims_l64_A3000_n30_nu30
[ "$SUMMARY_SET" = v1 ] || OUT=${OUT}_${SUMMARY_SET}
mkdir -p logs $OUT
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/diffcmb:$(pwd)/scripts
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 RAYON_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export DUCC0_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1
export TF_ENABLE_ONEDNN_OPTS=0
T=${SLURM_ARRAY_TASK_ID}
if [ "$T" -lt 4 ]; then
  FIRST=$((100000 + T * 50000)); N=50000; NAME=train_$T
else
  FIRST=500000; N=10000; NAME=heldout
fi
echo "task $T: realizations $FIRST..$((FIRST + N - 1)) -> $OUT/$NAME.npz"
.venv/bin/python -u scripts/npe/npe_simulate.py --first $FIRST --n $N --summary_set $SUMMARY_SET \
    --out $OUT/$NAME.npz
rc=$?
echo "task $T exited with status $rc"
exit $rc
