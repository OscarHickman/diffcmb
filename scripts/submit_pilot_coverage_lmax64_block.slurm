#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J pilot_coverage_lmax64_block
#SBATCH -o logs/pilot_coverage_lmax64_block_%j.out
#SBATCH -e logs/pilot_coverage_lmax64_block_%j.err
#SBATCH --exclusive

# Primary track, queued 2026-08-18 per ROADMAP.md's post-coupling-finding
# reprioritization (achievements.md, 2026-08-17: significant cross-L
# Hessian coupling confirmed in the phi-block posterior -- diagnosis job
# 11774832, mean coupling ratios 0.263/0.621 between L-bins [10,30) and
# [60,lmax), up to 0.97 at individual chain points). Every diagonal-in-L
# mass matrix tried so far (phi_mass_matrix='prior'/'fisher', any sampler,
# any scale, any tuning axis) failed the equilibration gate -- this job
# runs the first structurally different candidate: phi_mass_matrix='block'
# (lensing.py::estimate_phi_block_hessian, samplers.py::
# build_phi_block_mass_chol/PhiWhitener), a per-m-block Nystrom
# cross-L Hessian correction implemented and unit-tested 2026-08-18. Block
# mode is a one-time burn-in estimate (frozen thereafter) and is not
# supported together with Block 4 (C_L^phiphi|phi resampling every sweep)
# -- pilot_coverage_equilibration.py forces Block 4 off automatically when
# --phi_mass_matrix=block, unlike every prior gate run on this scale
# (pilot_coverage_lmax64_hmc_extended/lmax64_nuts), which had Block 4 on.
# Same lmax=64/nside=64 pilot scale and phi_hmc_step_size as the HMC
# baseline for a like-for-like comparison; phi_sampler stays 'hmc' (not
# 'nuts'/'mclmc') to isolate the mass-matrix change as the only variable
# from the validated HMC baseline.
LMAX=64
NSIDE=64
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

echo "Block-mass-matrix phi-block pilot: lmax=${LMAX} (nside=${NSIDE}), n_samples=${SAMPLES}, phi_mass_matrix=${PHI_MASS_MATRIX}, phi_block_n_probes=${PHI_BLOCK_N_PROBES}"
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
    --checkpoint_path results/analysis/pilot_coverage_lmax64_block_ckpt.npz \
    --checkpoint_every 50 \
    --out results/analysis/pilot_coverage_lmax64_block_n3300.npz
