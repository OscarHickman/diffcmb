#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -J diagnose_phi_hessian_coupling
#SBATCH -o logs/diagnose_phi_hessian_coupling_%j.out
#SBATCH -e logs/diagnose_phi_hessian_coupling_%j.err

# ROADMAP.md NEXT SESSION (2026-08-17): fresh-hypothesis diagnosis after all
# tuning-axis attempts at the phi-block equilibration prerequisite failed
# (achievements.md "Coverage-test prerequisite findings"). Tests whether
# cross-L Hessian coupling -- already hinted at in
# lensing.py::estimate_phi_diag_fisher's docstring (an abandoned Hutchinson
# estimator found large off-diagonal coupling, never quantified or connected
# to the equilibration failures) -- explains why every diagonal-in-L mass
# matrix (prior-only and one-shot Fisher, both) has failed regardless of
# tuning. No new MCMC: reuses the already-completed lmax=64 chain
# (job 11759467) and probes analytic-gradient finite differences at a few
# of its saved states. Cheap: submitted on durham/dine2 (148/200 durham jobs
# in use as of 2026-08-17 -- dp004 is now also at its 200-job cap, contrary
# to the stale note in ~/.claude/CLAUDE.md that it was "well under" -- update
# that note after this job clears).
#
# Used to decide between the two remaining untried structural fixes
# (dual-averaging/adaptive sampler vs. a non-diagonal mass matrix) rather
# than guessing -- see ROADMAP.md NEXT SESSION block.

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
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb:$(pwd)/scripts
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "phi-block Hessian cross-L coupling diagnosis (lmax=64, seed=0, from job 11759467's chain)"
echo "Using Python: $($PYTHON --version 2>&1)"

$PYTHON -u scripts/diagnose_phi_hessian_coupling.py \
    --chain_npz results/analysis/pilot_coverage_lmax64_hmc_extended_n6600.npz \
    --lmax 64 \
    --nside 64 \
    --n_chain_points 4 \
    --n_probes 6 \
    --out results/analysis/diagnose_phi_hessian_coupling.npz
