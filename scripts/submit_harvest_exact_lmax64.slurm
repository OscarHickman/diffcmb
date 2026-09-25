#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -J harvest_exact_l64
#SBATCH -o logs/harvest_exact_l64_%j.out
#SBATCH -e logs/harvest_exact_l64_%j.err
# 2026-09-23 harvest of the exact-operator ensembles (ROADMAP T0.2): strict
# C_L^phiphi SBC, Block 4 exactness PIT (with lag controls that must fail),
# phi power bias and coverage nulls, and the joint-likelihood SBC (replayed
# through each chain's own recorded operator and noise) -- both ensembles.
# results/analysis is a symlink into /cosma5, which dine2 does not mount (ROADMAP
# 2026-09-24). Fail loudly here rather than at the first write.
if [ ! -d /cosma/apps/durham/dc-hick2/diffcmb/results/analysis/ ]; then
    echo "results/analysis (/cosma5) is not mounted on $(hostname); relocate outputs before running on dine2" >&2
    exit 1
fi
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
for D in ens_exact_l64_A3000_n30 ens_exact_l64_A3000_n30_nocl4; do
  echo "================ $D: validate_coverage_rank_nulls (thin 10) ================"
  .venv/bin/python -u scripts/validate_coverage_rank_nulls.py \
      --indir results/analysis/$D --thin 10
  echo "================ $D: sbc_joint_likelihood (thin 10) ================"
  .venv/bin/python -u scripts/sbc_joint_likelihood.py \
      --indir results/analysis/$D --thin 10 \
      --out results/analysis/$D/sbc_joint_likelihood_thin10.npz
done
rc=$?
echo "Exited with status $rc"
# Propagate the status so sacct shows FAILED rather than COMPLETED.
exit $rc
