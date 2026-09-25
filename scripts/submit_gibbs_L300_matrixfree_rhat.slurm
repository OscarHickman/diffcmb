#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=6:00:00
#SBATCH -J cmb_gibbs_L300_mf_rhat
#SBATCH -o logs/cmb_gibbs_L300_mf_rhat_%j_%a.out
#SBATCH -e logs/cmb_gibbs_L300_mf_rhat_%j_%a.err
#SBATCH --array=1-4
#SBATCH --exclusive

# ROADMAP.md Section 0, next unchecked item: multi-chain R-hat test for the
# HMC + matrix-free-SHT pivot. Single-chain accuracy already validated
# (matches the Phase 0 dense-SHT reference to 1-3% across all l with these
# same map_steps/n_lfs/map_lr settings); this is the first genuine
# independent-chain R-hat check of that pivot before any production chain.
SAMPLER="gibbs"
LMAX=300
NSIDE=256
SAMPLES=3000
BURNIN=1500
STEP_SIZE=0.01
N_LFS=20
DATA_MODE="real"
DATA_DIR="/cosma8/data/dp004/dc-hick2/Plank"
OUTPUT_DIR="$(pwd)/results/lmax${LMAX}_nside${NSIDE}_${SAMPLER}_${DATA_MODE}_matrixfree_rhat"

# results/analysis is a symlink into /cosma5, which dine2 does not mount (ROADMAP
# 2026-09-24). Fail loudly here rather than at the first write.
if [ ! -d /cosma/apps/durham/dc-hick2/diffcmb/results/analysis/ ]; then
    echo "results/analysis (/cosma5) is not mounted on $(hostname); relocate outputs before running on dine2" >&2
    exit 1
fi
mkdir -p logs
mkdir -p $OUTPUT_DIR

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

echo "Starting Gibbs L=300 matrix-free-SHT multi-chain R-hat run (L=${LMAX}, NSIDE=${NSIDE})"
echo "Chain ID: ${SLURM_ARRAY_TASK_ID}"
echo "Using Python: $($PYTHON --version 2>&1)"
free -h

$PYTHON -u scripts/run_sampler.py \
    --sampler $SAMPLER \
    --lmax $LMAX \
    --nside $NSIDE \
    --n_samples $SAMPLES \
    --n_burnin $BURNIN \
    --step_size $STEP_SIZE \
    --n_lfs $N_LFS \
    --data_mode $DATA_MODE \
    --data_dir $DATA_DIR \
    --map_steps 2000 \
    --map_lr 0.01 \
    --use_matrixfree_sht \
    --chain_id $SLURM_ARRAY_TASK_ID \
    --output_dir $OUTPUT_DIR \
    --double_precision
