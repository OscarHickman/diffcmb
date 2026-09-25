#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J pilot_coverage_lmax128_block
#SBATCH -o logs/pilot_coverage_lmax128_block_%j.out
#SBATCH -e logs/pilot_coverage_lmax128_block_%j.err
#SBATCH --exclusive

# Scale-up test of the block-mass-matrix equilibration fix, queued 2026-08-18
# per ROADMAP.md's "Next" step after job 11781626 (lmax=64, phi_mass_matrix=
# 'block') passed the equilibration gate (worst lag-1 0.557, worst drift
# 0.20 sigma -- achievements.md). That result confirms the per-m-block
# Nystrom cross-L Hessian correction (lensing.py::estimate_phi_block_hessian,
# samplers.py::build_phi_block_mass_chol/PhiWhitener) fixes equilibration at
# lmax=64, but the original diagonal-mass-matrix failures were themselves
# scale-dependent (the harmonic band-edge hypothesis was tested and
# falsified specifically at lmax=128, achievements.md) -- so lmax=64 passing
# does not by itself guarantee lmax=128 passes. This run re-checks the same
# gate at the real coverage-test resolution before trusting the fix for the
# lmax~100-150 rank/coverage ensemble (ROADMAP.md item 1).
#
# Same n_samples/n_lfs/phi_n_lfs/phi_mass_matrix budget as the lmax=64 GO
# run (submit_pilot_coverage_lmax64_block.slurm) for a like-for-like
# comparison; only lmax/nside change. Checkpointed every 50 sweeps
# (results/analysis/pilot_coverage_lmax128_block_ckpt.npz) so a 24h walltime
# cutoff mid-chain can be resumed with a follow-up submission pointing at
# the same checkpoint path, same pattern as
# submit_pilot_coverage_lmax128_phi240.slurm/_extend.slurm.
LMAX=128
NSIDE=128
BURNIN=400
SAMPLES=3300
N_LFS=10
PHI_SAMPLER=hmc
PHI_HMC_STEP_SIZE=0.05
PHI_N_LFS=240
PHI_MASS_MATRIX=block
PHI_FISHER_WARMUP_ITER=50
PHI_FISHER_N_PROBES=8
PHI_BLOCK_N_PROBES=6
MAP_STEPS=2000
MAP_LR=0.01

# results/analysis is a symlink into /cosma5, which dine2 does not mount (ROADMAP
# 2026-09-24). Fail loudly here rather than at the first write.
if [ ! -d /cosma/apps/durham/dc-hick2/diffcmb/results/analysis/ ]; then
    echo "results/analysis (/cosma5) is not mounted on $(hostname); relocate outputs before running on dine2" >&2
    exit 1
fi
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

mkdir -p logs results/analysis

export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "Block-mass-matrix phi-block pilot, scale-up test: lmax=${LMAX} (nside=${NSIDE}), n_samples=${SAMPLES}, phi_mass_matrix=${PHI_MASS_MATRIX}, phi_block_n_probes=${PHI_BLOCK_N_PROBES}"
echo "Using Python: $($PYTHON --version 2>&1)"
free -h

$PYTHON -u scripts/pilot_coverage_equilibration.py \
    --lmax $LMAX \
    --nside $NSIDE \
    --n_burnin $BURNIN \
    --map_steps $MAP_STEPS \
    --map_lr $MAP_LR \
    --n_samples $SAMPLES \
    --n_lfs $N_LFS \
    --phi_sampler $PHI_SAMPLER \
    --phi_hmc_step_size $PHI_HMC_STEP_SIZE \
    --phi_n_lfs $PHI_N_LFS \
    --phi_mass_matrix $PHI_MASS_MATRIX \
    --phi_fisher_warmup_iter $PHI_FISHER_WARMUP_ITER \
    --phi_fisher_n_probes $PHI_FISHER_N_PROBES \
    --phi_block_n_probes $PHI_BLOCK_N_PROBES \
    --checkpoint_path results/analysis/pilot_coverage_lmax128_block_ckpt.npz \
    --checkpoint_every 50 \
    --out results/analysis/pilot_coverage_lmax128_block_n3300.npz
