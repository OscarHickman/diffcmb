#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J pilot_coverage_lmax128_block_resume
#SBATCH -o logs/pilot_coverage_lmax128_block_resume_%j.out
#SBATCH -e logs/pilot_coverage_lmax128_block_resume_%j.err
#SBATCH --exclusive

# Resume of job 11795998 (submit_pilot_coverage_lmax128_block.slurm), which
# hit the 24h walltime with 1150/3300 samples collected (checkpoint:
# results/analysis/pilot_coverage_lmax128_block_ckpt.npz). run_gibbs_chain's
# resume path (diffcmb/samplers.py) restores full chain state and forces
# burnin_remaining=0, so this picks up exactly where 11795998 left off with
# n_samples_remaining = 3300 - 1150 = 2150.
#
# Observed rate from 11795998: ~1550 sweeps (400 burn-in + 1150 samples) in
# ~24h => ~56s/sweep. At that rate 2150 remaining sweeps is ~33.4h, so this
# will likely hit walltime again before finishing -- expect a further
# resubmission against the same checkpoint (same pattern as
# submit_pilot_coverage_lmax128_extend.slurm's precedent). Checkpointed
# every 50 sweeps so no progress is lost on a repeat timeout.
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

echo "Resuming block-mass-matrix phi-block pilot at lmax=${LMAX} (nside=${NSIDE}) from checkpoint (1150/${SAMPLES} already collected)"
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
