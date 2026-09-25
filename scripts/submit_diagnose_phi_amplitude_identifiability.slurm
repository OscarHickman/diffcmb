#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska,bluefield1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -J diag_phi_amp_ident
#SBATCH -o logs/diag_phi_amp_ident_%j.out
#SBATCH -e logs/diag_phi_amp_ident_%j.err
# Not dine2: results/analysis is a symlink into /cosma5, which dine2 does not mount
# (ROADMAP 2026-09-24). Memory sized from past MaxRSS instead of a whole node.

# Root-cause diagnostic for the 2026-08-28 coverage-ensemble failure
# (job 11887897): phi and C_L^phiphi ranked 0/8 in every ell-bin of every
# realization, with phi frozen ~7e4x above truth in power.
#
# Evidence already established offline from the saved chains (no new compute):
#   * the chain START is at the correct phi scale, so burn-in drove it up;
#   * phi power is stable to ~2% across all 600 saved samples where ~50%
#     scatter is expected -- the block is frozen, not sampling;
#   * logp moves monotonically across the whole "production" window with no
#     plateau, so burn-in never finished;
#   * alm-vs-truth cosine similarity is only 0.39-0.46 AND DECREASING;
#   * every lmax=64 PILOT (which runs a MAP pre-optimisation the ensemble
#     script omits) sits at the correct phi scale, ratio 0.09-2.5.
#
# This job settles the one remaining fork, analytically motivated in the
# script header: with Block 4 on, the refitted C_L^phiphi makes the phi prior
# term EXACTLY scale-free, so the amplitude is pinned by the lensing
# likelihood alone. Either that likelihood has a minimum at the true
# amplitude (=> the ensemble failure is a warm-start bug, fix = MAP + longer
# burn-in) or it does not (=> Block 4 is ill-posed at this configuration and
# no warm start rescues it). Forward evaluations only, no MCMC -- minutes.

set -u
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
mkdir -p logs results/analysis
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
PYTHON=".venv/bin/python"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0

echo "phi amplitude identifiability diagnostic (lmax=64, nside=64)"
echo "Using Python: $($PYTHON --version 2>&1)"

$PYTHON -u scripts/diagnose_phi_amplitude_identifiability.py \
    --lmax 64 --nside 64 --realization 0

rc=$?
echo "Exited with status $rc"
# Propagate the status so sacct shows FAILED rather than COMPLETED.
exit $rc
