#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma8-shm,cosma8-ska,cosma8-shm2,cosma8-shm3,cosma5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH -J harvest_t01c
#SBATCH -o logs/harvest_t01c_%j.out
#SBATCH -e logs/harvest_t01c_%j.err
# ROADMAP T0.1 (c2, c3): harvest of the long Block-4-ON chains (1000 burn-in +
# 3600 samples) at nu = 6 (job 12047785) and nu = 30 (job 12047786), same skies
# as the 1200-sweep ensembles. Per ensemble: stationarity by thin/half/quarter,
# joint-likelihood SBC (thin 40, and 2nd half at thin 10), the strict C_L^phiphi
# SBC + Block 4 PIT + coverage nulls, and the a_lm power-rank null.
# Not dine2: results/analysis is a symlink into /cosma5, which dine2 does not mount.
set -u
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
mkdir -p logs
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/diffcmb
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 RAYON_NUM_THREADS=16 NUMBA_NUM_THREADS=16
export TF_NUM_INTRAOP_THREADS=16 TF_NUM_INTEROP_THREADS=1
export TF_ENABLE_ONEDNN_OPTS=0
LONG=results/analysis/ens_exact_l64_A3000_n30_long
NU30=results/analysis/ens_exact_l64_A3000_n30_nu30_long
rc=0
run() { "$@" || { echo "FAILED ($?): $*"; rc=1; }; }
run .venv/bin/python -u scripts/diagnose_calibration_stationarity.py \
    --indir $LONG --indir $NU30 --thins 10,40 --n_seg 4
for D in $LONG $NU30; do
  echo "================ $D: sbc_joint_likelihood (thin 40) ================"
  run .venv/bin/python -u scripts/sbc_joint_likelihood.py --indir $D --thin 40 \
      --out $D/sbc_joint_likelihood_thin40.npz
  echo "================ $D: sbc_joint_likelihood (second half, thin 10) ================"
  run .venv/bin/python -u scripts/sbc_joint_likelihood.py --indir $D --thin 10 \
      --burn_frac 0.5 --out $D/sbc_joint_likelihood_thin10_half2.npz
  echo "================ $D: validate_coverage_rank_nulls (thin 10) ================"
  run .venv/bin/python -u scripts/validate_coverage_rank_nulls.py --indir $D --thin 10
done
echo "================ a_lm power-rank null ================"
run .venv/bin/python -u scripts/null_alm_power_rank_flat_prior.py \
    --indir $LONG --indir $NU30 --n_null 2000
echo "Exited with status $rc"
exit $rc
