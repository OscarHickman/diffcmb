#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J cov_ens_l64_prior_nocl4_packingv2
#SBATCH -o logs/cov_ens_l64_prior_nocl4_packingv2_%A_%a.out
#SBATCH -e logs/cov_ens_l64_prior_nocl4_packingv2_%A_%a.err
#SBATCH --exclusive
#SBATCH --array=0-11

# ===================================================================
# 2026-09-08 12-REALIZATION RE-RUN OF THE HEADLINE EXACTNESS ENSEMBLE
# UNDER THE RESTORED 2L+1 PACKING.
#
# job 11903181 (12 realizations, 2L packing, Block 4 OFF) produced the
# headline pair phi 0.4688 (KS_p 0.124) / alm 0.5312 (KS_p 0.235), both
# uniform -- but that packing forced Im(a_{L,1}) = 0, so it could not
# represent a general sky (achievements.md, docs/notes/
# restore_missing_alm_dof_scoping.md). The dof fix landed 2026-09-06
# (629c0ee) and a single-realization pilot (job 11951115) validated the
# new packing at production scale:
#   - phi-power/truth ratio 0.9612 (O(1))
#   - alm-vs-truth cosine similarity +0.9942 (high)
#   - logp plateaued: mean 26200.5 (sweeps 0-299) vs 26198.0 (300-599),
#     flat within noise
# That clears the gate ROADMAP.md set before this ensemble could go out.
#
# DELIBERATELY the same configuration as job 11903181 / the pilot:
# lmax=64, nside=64, phi_mass_matrix='prior', n_burnin=100, 600 samples,
# n_lfs=10, phi_n_lfs=240, Block 4 OFF (--no_sample_cl_phiphi, so
# cl_phiphi_full stays pinned at the fiducial and the phi prior is proper
# and identical to the generative process -- the valid-SBC configuration).
# The packing restoration is the only intended difference from 11903181.
#
# Truth/data/start draws are NOT bit-identical to 11903181's per-realization
# ones even at the same --realization index and seed streams, because
# hp.synalm now keeps the Im(a_{L,1}) it used to discard (same note as the
# pilot script). This is expected and is the point of the re-run.
#
# Cost: ~24.8s/sweep pre-restoration x 700 sweeps ~ 4.8h/realization; the
# packed vector grows by lmax-2=62 coordinates out of ~2080 (~3%), so budget
# ~5h/realization. 24h walltime, checkpoint every 50 sweeps.
#
# Job cap: 12 array tasks, re-checked before submitting (limit is 200):
#   squeue -u dc-hick2 --noheader -r -o "%a" | grep -c durham
#
# HARVEST. Check .err for tracebacks per task (a COMPLETED state alone is
# not enough -- achievements.md's return-tuple-arity entry), confirm each
# realization's phi_power_ratio_to_truth is O(1) and phi_calibration_ok is
# True, then:
#   PYTHONPATH=diffcmb .venv/bin/python scripts/aggregate_coverage_ranks.py \
#       --indir results/analysis/coverage_ensemble_lmax64_prior_nocl4_packingv2 \
#       --thin 90
# Re-derive --thin from this run's own tau_int rather than assuming 90
# carries over unchanged. No cl_phiphi row by construction (Block 4 off).
# Compare the resulting phi/alm pooled mean_u and KS_p directly against the
# 2L reference (phi 0.4688/p=0.124, alm 0.5312/p=0.235, job 11903181) --
# that comparison is the whole point of this ensemble.
# ===================================================================

LMAX=64
NSIDE=64
BURNIN=100
SAMPLES=600
N_LFS=10
PHI_N_LFS=240

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

mkdir -p logs results/analysis/coverage_ensemble_lmax64_prior_nocl4_packingv2
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "Coverage ensemble (packing v2, 2L+1 dof) realization ${SLURM_ARRAY_TASK_ID} at lmax=${LMAX} (prior mass matrix, Block 4 OFF)"
echo "git HEAD: $(git rev-parse --short HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Using Python: $($PYTHON --version 2>&1)"

$PYTHON -u scripts/coverage_ensemble_chain.py \
    --realization ${SLURM_ARRAY_TASK_ID} \
    --lmax $LMAX \
    --nside $NSIDE \
    --n_burnin $BURNIN \
    --n_samples $SAMPLES \
    --n_lfs $N_LFS \
    --phi_n_lfs $PHI_N_LFS \
    --phi_mass_matrix prior \
    --no_sample_cl_phiphi \
    --outdir results/analysis/coverage_ensemble_lmax64_prior_nocl4_packingv2 \
    --checkpoint_every 50

echo "Realization ${SLURM_ARRAY_TASK_ID} exited with status $?"
