#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J cov_ens_l64
#SBATCH -o logs/cov_ens_l64_%A_%a.out
#SBATCH -e logs/cov_ens_l64_%A_%a.err
#SBATCH --exclusive
#SBATCH --array=0-11

# ROADMAP.md Section 1, Priority 1 -- THE headline validation, launched at the
# ONE configuration that has ever cleared the phi equilibration gate.
#
# Configuration provenance (do not change without a new passing pilot):
#   lmax=64, nside=64, phi_n_lfs=240, phi_mass_matrix='block', Block 4 OFF
#   == job 11781626, harvested 2026-08-18: GO. Worst lag-1 autocorrelation
#      0.557 (gate <0.9), worst drift 0.20 sigma (gate <2.0).
#
# Why Block 4 is OFF here (--no_sample_cl_phiphi): job 11836793 gated the
# Block-4-ON version of this exact configuration on 2026-08-24 and it was
# NO-GO -- worst lag-1 0.945. Turning Block 4 on degrades phi mixing past the
# gate even at lmax=64. Consequence: this ensemble delivers the SBC-supported
# field-level ranks (phi_power, alm_power) and C_l^TT interval coverage, but
# NOT the C_L^phiphi coverage row. That row waits on the non-centred
# reparameterisation work; it is not a defect of this run.
#
# Why lmax=64 and not 128: the lmax=128 scale-up is NO-GO (job 11830353, worst
# lag-1 0.996 at low L). ROADMAP.md's standing rule -- "prefer a converged
# result at a smaller scale over a non-converged one at a larger scale, every
# time" -- makes 64 the correct spend. Job 11781626's own verdict line
# recommended exactly this.
#
# Cost: pilot measured ~12s/sweep at this configuration, so 100 burn-in + 600
# samples ~ 2.4h per realization -- well inside one 24h walltime. Each task
# checkpoints every 50 sweeps regardless.
#
# Job cap: 12 array tasks. Re-check before submitting (limit is 200):
#   squeue -u dc-hick2 --noheader -r -o "%a" | grep -c durham
#
# After all tasks finish, aggregate with --thin set from the pilot's measured
# lag-k autocorrelation (first lag with |r_k|<0.2 was 25 for the worst bin):
#   PYTHONPATH=diffcmb .venv/bin/python scripts/aggregate_coverage_ranks.py \
#       --indir results/analysis/coverage_ensemble_lmax64 --thin 25
LMAX=64
NSIDE=64
BURNIN=100
SAMPLES=600
N_LFS=10
PHI_N_LFS=240

# results/analysis is a symlink into /cosma5, which dine2 does not mount (ROADMAP
# 2026-09-24). Fail loudly here rather than at the first write.
if [ ! -d /cosma/apps/durham/dc-hick2/diffcmb/results/analysis/ ]; then
    echo "results/analysis (/cosma5) is not mounted on $(hostname); relocate outputs before running on dine2" >&2
    exit 1
fi
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

mkdir -p logs results/analysis/coverage_ensemble_lmax64
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "Coverage ensemble realization ${SLURM_ARRAY_TASK_ID} at lmax=${LMAX} (block mass matrix, Block 4 OFF)"
echo "Using Python: $($PYTHON --version 2>&1)"

$PYTHON -u scripts/coverage_ensemble_chain.py \
    --realization ${SLURM_ARRAY_TASK_ID} \
    --lmax $LMAX \
    --nside $NSIDE \
    --n_burnin $BURNIN \
    --n_samples $SAMPLES \
    --n_lfs $N_LFS \
    --phi_n_lfs $PHI_N_LFS \
    --phi_mass_matrix block \
    --no_sample_cl_phiphi \
    --outdir results/analysis/coverage_ensemble_lmax64 \
    --checkpoint_every 50

rc=$?
echo "Realization ${SLURM_ARRAY_TASK_ID} exited with status $rc"
# Propagate the status so sacct shows FAILED rather than COMPLETED.
exit $rc
