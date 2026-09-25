#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J pilot_coverage_lmax64_block_cl4
#SBATCH -o logs/pilot_coverage_lmax64_block_cl4_%j.out
#SBATCH -e logs/pilot_coverage_lmax64_block_cl4_%j.err
#SBATCH --exclusive

# Production-configuration gate, queued 2026-08-23 per ROADMAP.md's
# "Blocker found 2026-08-23" entry: job 11781626 (submit_pilot_coverage_
# lmax64_block.slurm) gated phi_mass_matrix='block' at lmax=64 with Block 4
# (C_L^phiphi|phi) forced OFF -- GO (worst lag-1 autocorr 0.557, worst
# drift 0.20 sigma) -- but Block 4 is what produces the paper's headline
# joint (C_l, C_L^phiphi) posterior figure, and until 2026-08-23 Block 4
# was mutually exclusive with phi_mass_matrix='block' in samplers.py::
# run_gibbs_chain (a conservative efficiency guard, not a correctness
# requirement -- see samplers.py's run_gibbs_chain docstring and
# achievements.md/ROADMAP.md for the argument). That guard was relaxed
# this session (commit 4f6eb74): the expensive likelihood-curvature
# estimate (diag_fisher_per_L/block_hessian) is frozen at burn-in as
# before, but the diagonal prior-precision term is now cheaply rebuilt
# every sweep to track Block 4's resampled spectrum.
#
# This run gates that ACTUAL production configuration for the first time
# (Block 4 ON + phi_mass_matrix='block') at the one scale that has ever
# passed the diagonal-mass-matrix-free equilibration gate. Same lmax=64/
# nside=64/burn-in/samples/step-size/n_lfs as job 11781626 for a direct
# like-for-like comparison -- the only change is --phi_mass_matrix's
# Block-4 interaction, now auto-enabled by pilot_coverage_equilibration.py
# (use_block4 = args.phi_mass_matrix in ('prior', 'block')).
#
# Known risk flagged before launch (ROADMAP.md): job 11781626's GO was
# measured with Block 4 off. Turning it on couples low-L phi amplitudes to
# a C_L^phiphi drawn from an inverse-Gamma with very few modes (L=2 has 5)
# -- a centred-parameterisation funnel, exactly the geometry the lmax=128
# NO-GO (job 11830353) already showed is the weak spot. This run is
# designed to surface that risk at a scale that's cheap to iterate on
# (~11h), not to assume it away.
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

echo "Block-mass-matrix + Block 4 (C_L^phiphi|phi) phi-block pilot: lmax=${LMAX} (nside=${NSIDE}), n_samples=${SAMPLES}, phi_mass_matrix=${PHI_MASS_MATRIX}, phi_block_n_probes=${PHI_BLOCK_N_PROBES}"
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
    --checkpoint_path results/analysis/pilot_coverage_lmax64_block_cl4_ckpt.npz \
    --checkpoint_every 50 \
    --out results/analysis/pilot_coverage_lmax64_block_cl4_n3300.npz
