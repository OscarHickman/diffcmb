#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J pilot_coverage_lmax64_block_cl4_ordfix
#SBATCH -o logs/pilot_coverage_lmax64_block_cl4_ordfix_%j.out
#SBATCH -e logs/pilot_coverage_lmax64_block_cl4_ordfix_%j.err
#SBATCH --exclusive

# Re-run of submit_pilot_coverage_lmax64_block_cl4.slurm (job 11836793),
# POST the alm-ordering fix (commit a16e9e6, 2026-08-24 20:08). Job
# 11836793 ran 2026-08-23 12:23 -> 2026-08-24 03:50 -- entirely BEFORE the
# fix landed, despite being harvested after, so its result is invalid
# (achievements.md's "post-ordering-fix baseline" entry). This is the only
# untested-post-fix phi mass matrix, and the paper's actual production
# configuration (lmax=64, phi_mass_matrix='block', Block 4 ON).
#
# Motivation (ROADMAP.md, achievements.md, 2026-08-26): the ONE valid
# post-fix phi run so far (job 11849969, plain 'prior' diagonal mass
# matrix, re-scored with scripts/rescore_phi_equilibration.py instead of
# the raw-lag-1 gate) is stationary and usable in 3 of 4 l-bins; the sole
# failure is the top band-edge bin [60,64) (split-Rhat 1.404, tau_int 326).
# 'block' is the one mechanism aimed at exactly that kind of cross-L /
# band-edge coupling. If it clears the (corrected) gate here, the lmax=64
# coverage ensemble can launch on this exact configuration and deliver both
# the exactness evidence and the headline joint (C_l, C_L^phiphi) posterior
# figure from one run.
#
# Identical sampler config to job 11836793 (lmax=64/nside=64/burn-in=400/
# samples=3300/step-size/n_lfs/phi_mass_matrix=block/Block4 ON) -- only
# the code (ordering fix) and output/checkpoint paths differ, to avoid
# overwriting the (invalid but still-referenced) pre-fix result file.
#
# Score this with scripts/rescore_phi_equilibration.py (tau_int/ESS/
# split-Rhat), NOT the raw lag-1 < 0.9 gate -- that gate was shown to be a
# measurement artifact for the 11849969 run (achievements.md).
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

echo "Post-ordering-fix block-mass-matrix + Block 4 (C_L^phiphi|phi) phi-block pilot: lmax=${LMAX} (nside=${NSIDE}), n_samples=${SAMPLES}, phi_mass_matrix=${PHI_MASS_MATRIX}, phi_block_n_probes=${PHI_BLOCK_N_PROBES}"
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
    --checkpoint_path results/analysis/pilot_coverage_lmax64_block_cl4_ordfix_ckpt.npz \
    --checkpoint_every 50 \
    --out results/analysis/pilot_coverage_lmax64_block_cl4_ordfix_n3300.npz
