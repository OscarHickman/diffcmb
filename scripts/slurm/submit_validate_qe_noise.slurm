#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska,bluefield1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -J qe_noise_val
#SBATCH -o logs/qe_noise_val_%j.out
#SBATCH -e logs/qe_noise_val_%j.err
# Not dine2: results/analysis is a symlink into /cosma5, which dine2 does not mount
# (ROADMAP 2026-09-24). Memory sized from past MaxRSS instead of a whole node.

# 2026-09-18. Validates diffcmb/qe.py's analytic N_L^phiphi against the
# own definition by Monte Carlo (scripts/validate_qe_noise.py). This is the
# gate that licenses paper figure 3's QE comparison curve -- plots/STORY.md
# forbids that curve appearing in a figure until this passes.

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
mkdir -p logs results/analysis
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=16
export TF_ENABLE_ONEDNN_OPTS=0

.venv/bin/python -u scripts/validate_qe_noise.py \
    --lmax 64 --nside 64 --noisesig 1.0 \
    --n_sims 64 \
    --out results/analysis/qe_noise_validation.npz

rc=$?
echo "Exited with status $rc"
# Propagate the status so sacct shows FAILED rather than COMPLETED.
exit $rc
