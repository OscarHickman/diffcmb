#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH -J diag_calib_exact
#SBATCH -o logs/diag_calib_exact_%j.out
#SBATCH -e logs/diag_calib_exact_%j.err
# ROADMAP T0.1 (a, b): is the exact-operator calibration failure thinning,
# burn-in, or the sampler? Field ranks at thin 10/40 by chain half, rank mean
# per chain quarter, drift z of the power traces (numpy only), then the
# joint-likelihood SBC at thin 40 and on the second half at thin 10.
set -u
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
mkdir -p logs
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/diffcmb
export OMP_NUM_THREADS=16
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
ON=results/analysis/ens_exact_l64_A3000_n30
OFF=results/analysis/ens_exact_l64_A3000_n30_nocl4
.venv/bin/python -u scripts/diagnose_calibration_stationarity.py \
    --indir $ON --indir $OFF --thins 10,40 --n_seg 4
for D in $ON $OFF; do
  echo "================ $D: sbc_joint_likelihood (thin 40) ================"
  .venv/bin/python -u scripts/sbc_joint_likelihood.py --indir $D --thin 40 \
      --out $D/sbc_joint_likelihood_thin40.npz
  echo "================ $D: sbc_joint_likelihood (second half, thin 10) ================"
  .venv/bin/python -u scripts/sbc_joint_likelihood.py --indir $D --thin 10 \
      --burn_frac 0.5 --out $D/sbc_joint_likelihood_thin10_half2.npz
done
echo "Exited with status $?"
