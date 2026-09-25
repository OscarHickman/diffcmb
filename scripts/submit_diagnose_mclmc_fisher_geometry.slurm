#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH -J diagnose_mclmc_fisher_geometry
#SBATCH -o logs/diagnose_mclmc_fisher_geometry_%j.out
#SBATCH -e logs/diagnose_mclmc_fisher_geometry_%j.err
#SBATCH --exclusive

# ROADMAP.md "Sampler-lever decision", trigger 3(a)/(b) reached 2026-08-14
# (job 11744132: MCLMC NO-GO at lmax=128 with the tuned prior-only mass
# matrix -- worst lag-1=1.000, worst drift=6.55 sigma across ALL 5 l-bins).
# User decision 2026-08-14: pursue a geometry fix (this job) AND a
# lower-scale fallback in parallel (submit_pilot_coverage_lmax64_hmc.slurm).
#
# Geometry hypothesis: lensing.py::estimate_phi_diag_fisher's own docstring
# (written 2026-08-08 for the HMC path, never tested against MCLMC) predicts
# exactly this failure mode -- "the whitened posterior variance is highly
# non-uniform across L ... so a single global step size settles at whatever
# the tightest dimension needs, moving almost nowhere in the rest." A
# Fisher-informed (posterior-curvature) mass matrix was tried for HMC and
# closed as unfavorable (achievements.md), but never tried for MCLMC, whose
# isokinetic dynamics are fundamentally different from HMC's momentum-based
# dynamics -- the HMC closure is not strong evidence this also fails for
# MCLMC. Testing it is the cheapest available geometry fix: the fisher-warmup
# code path (samplers.py, phi_fisher_warmup_iter/phi_fisher_n_probes) is
# already implemented, validated by unit test
# (test_gibbs_chain_phi_mass_matrix_fisher_moves), and generic across both
# integrators -- only the guard blocking phi_sampler='mclmc' +
# phi_mass_matrix='fisher' needed relaxing (diffcmb/samplers.py, 2026-08-14).
#
# CAVEAT, not resolved by this job: the Fisher mass matrix is a one-time
# burn-in estimate, frozen thereafter -- incompatible with Block 4
# (sample_cl_phiphi=True) resampling cl_phiphi_full every sweep. This run
# uses scripts/diagnose_mclmc_fisher_geometry.py, a dedicated script with
# Block 4 OFF (fixed cl_phiphi_full at the true LCDM spectrum) -- a
# diagnostic of whether preconditioning fixes the geometry at all, NOT a
# directly-deployable production configuration even if GO. If GO,
# reconciling with Block 4 (e.g. periodic Fisher re-estimation instead of
# once-at-burn-in) is a separate follow-up, not yet scoped.
#
# Same step_size=0.005/L=100/n_steps=30 as the failed v2 pilot (job
# 11744132) -- deliberately unchanged, so any improvement is attributable to
# the mass matrix, not a second simultaneous change. Fresh checkpoint (not a
# resume of 11744132's, since the mass-matrix rescaling of the whitened phi
# state at burn-in makes an old checkpoint's state inconsistent with a fisher
# run from the start).
#
# n_samples=3300 matches every prior pilot's window for direct comparability.
# phi_fisher_warmup_iter=20 (well before n_burnin=400 ends) gives the chain
# time to move past the literal warm-start point before the curvature probe.
#
# Checkpoint on /cosma8 archive per the 2026-08-13 lesson (achievements.md):
# /cosma/apps is NFS-quota-shared across all dc-hick2 projects and caused a
# multi-submission failure loop on the v2 pilot; large checkpoints belong on
# /cosma8/data/dp004/dc-hick2/diffcmb_results_archive/, confirmed mounted on
# dine2 via srun, not /cosma5 (not mounted on dine2 compute nodes at all).
#
# Walltime budget: 48h. dine2's partition MaxTime is 3-00:00:00 (72h).
LMAX=128
NSIDE=128
BURNIN=400
SAMPLES=3300
N_LFS=10
MAP_STEPS=2000
MAP_LR=0.01
PHI_HMC_STEP_SIZE=0.005
PHI_N_LFS=30
PHI_MCLMC_L=100.0
PHI_FISHER_WARMUP_ITER=20
PHI_FISHER_N_PROBES=8

# results/analysis is a symlink into /cosma5, which dine2 does not mount (ROADMAP
# 2026-09-24). Fail loudly here rather than at the first write.
if [ ! -d /cosma/apps/durham/dc-hick2/diffcmb/results/analysis/ ]; then
    echo "results/analysis (/cosma5) is not mounted on $(hostname); relocate outputs before running on dine2" >&2
    exit 1
fi
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

mkdir -p logs results/analysis
ARCHIVE_DIR="/cosma8/data/dp004/dc-hick2/diffcmb_results_archive"
mkdir -p "$ARCHIVE_DIR"
CKPT_TARGET="${ARCHIVE_DIR}/diagnose_mclmc_fisher_geometry_ckpt.npz"
CKPT_LINK="results/analysis/diagnose_mclmc_fisher_geometry_ckpt.npz"
if [ ! -e "$CKPT_LINK" ]; then
    ln -s "$CKPT_TARGET" "$CKPT_LINK"
fi

export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb:$(pwd)/scripts
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "MCLMC + Fisher mass matrix geometry spike: lmax=${LMAX} (nside=${NSIDE}), step_size=${PHI_HMC_STEP_SIZE}, L=${PHI_MCLMC_L}, n_steps=${PHI_N_LFS}"
echo "Using Python: $($PYTHON --version 2>&1)"
free -h

$PYTHON -u scripts/diagnose_mclmc_fisher_geometry.py \
    --lmax $LMAX \
    --nside $NSIDE \
    --n_burnin $BURNIN \
    --map_steps $MAP_STEPS \
    --map_lr $MAP_LR \
    --n_samples $SAMPLES \
    --n_lfs $N_LFS \
    --phi_hmc_step_size $PHI_HMC_STEP_SIZE \
    --phi_n_lfs $PHI_N_LFS \
    --phi_mclmc_L $PHI_MCLMC_L \
    --phi_fisher_warmup_iter $PHI_FISHER_WARMUP_ITER \
    --phi_fisher_n_probes $PHI_FISHER_N_PROBES \
    --checkpoint_path "$CKPT_LINK" \
    --checkpoint_every 50 \
    --out results/analysis/diagnose_mclmc_fisher_geometry_n3300.npz
