#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH -J gate_phi_mclmc_vs_hmc_L128
#SBATCH -o logs/gate_phi_mclmc_vs_hmc_L128_%j.out
#SBATCH -e logs/gate_phi_mclmc_vs_hmc_L128_%j.err
#SBATCH --exclusive

# MCLMC spike (ROADMAP.md, 2026-08-07): Block 3 (phi | alm, C_l) validation
# gate at the lmax~128 pilot scale -- the small-lmax=20 grid (2026-08-08)
# found a config (n_steps=30, step_size=0.1, L=40) beating HMC on 4/6 probed
# l-bins with a clean bias check; this run tests whether that transfers to
# the actual pilot scale/problem bin (l=[60,100), lag-1=0.991 under HMC
# phi_n_lfs=240 -- see ROADMAP.md "Currently doing"). L is scaled by
# sqrt(d_128/d_20) approx 6.6x from the lmax=20 winner (40 -> ~260); tried at
# two nearby values since this scaling is a rule-of-thumb, not derived.
# PHI_N_LFS=240 matches the recorded best HMC configuration exactly, for a
# fair comparison.
LMAX=128
NSIDE=64
BURNIN=100
SAMPLES=200
PHI_N_LFS=240
MCLMC_N_STEPS="30"
MCLMC_STEP_SIZE="0.1"
MCLMC_L="200,350"

# results/analysis is a symlink into /cosma5, which dine2 does not mount (ROADMAP
# 2026-09-24). Fail loudly here rather than at the first write.
if [ ! -d /cosma/apps/durham/dc-hick2/diffcmb/results/analysis/ ]; then
    echo "results/analysis (/cosma5) is not mounted on $(hostname); relocate outputs before running on dine2" >&2
    exit 1
fi
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

mkdir -p logs results/analysis

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "Starting MCLMC spike gate: Block 3 phi HMC vs MCLMC (lmax=${LMAX}, nside=${NSIDE})"
echo "Using Python: $($PYTHON --version 2>&1)"
free -h

$PYTHON -u scripts/gate_phi_mclmc_vs_hmc.py \
    --lmax $LMAX \
    --nside $NSIDE \
    --n_burnin $BURNIN \
    --n_samples $SAMPLES \
    --phi_n_lfs $PHI_N_LFS \
    --mclmc_n_steps $MCLMC_N_STEPS \
    --mclmc_step_size "$MCLMC_STEP_SIZE" \
    --mclmc_L "$MCLMC_L" \
    --out results/analysis/phi_mclmc_vs_hmc_lmax${LMAX}_pilot.npz
