#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J pilot_coverage_lmax64_nuts
#SBATCH -o logs/pilot_coverage_lmax64_nuts_%j.out
#SBATCH -e logs/pilot_coverage_lmax64_nuts_%j.err
#SBATCH --exclusive

# Secondary/comparison track, queued 2026-08-18 per ROADMAP.md's post-coupling-
# finding reprioritization (achievements.md, 2026-08-17: significant cross-L
# Hessian coupling confirmed in the phi-block posterior -- diagnosis job
# 11774832). A block/low-rank mass matrix is now the primary fix candidate,
# not yet implemented. This job runs phi_sampler='nuts' (dynamic trajectory
# length, implemented/unit-tested 2026-08-17) at the same lmax=64 pilot scale
# as the HMC baseline (pilot_coverage_lmax64_hmc_extended, n6600) purely as a
# comparison point: NUTS still sits on a diagonal mass matrix, so it is NOT
# expected to clear the equilibration gate by itself, but the run establishes
# whether dynamic trajectory length alone buys any measurable improvement
# over fixed-schedule HMC on the same stuck bins, informing whether it's
# worth combining with the block mass matrix later.
LMAX=64
NSIDE=64
BURNIN=400
SAMPLES=3300
N_LFS=10
PHI_SAMPLER=nuts
PHI_HMC_STEP_SIZE=0.05
PHI_NUTS_MAX_TREE_DEPTH=8
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

echo "NUTS phi-block comparison pilot: lmax=${LMAX} (nside=${NSIDE}), n_samples=${SAMPLES}, phi_nuts_max_tree_depth=${PHI_NUTS_MAX_TREE_DEPTH}"
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
    --phi_nuts_max_tree_depth $PHI_NUTS_MAX_TREE_DEPTH \
    --checkpoint_path results/analysis/pilot_coverage_lmax64_nuts_ckpt.npz \
    --checkpoint_every 50 \
    --out results/analysis/pilot_coverage_lmax64_nuts_n3300.npz
