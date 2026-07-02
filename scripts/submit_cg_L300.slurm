#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH -J cmb_cg_L300
#SBATCH -o logs/cmb_cg_L300_%j_%a.out
#SBATCH -e logs/cmb_cg_L300_%j_%a.err
#SBATCH --array=1-4
#SBATCH --exclusive

# Phase 0b: CG-based exact alm sampler warm-started from Phase 0 HMC checkpoints.
# Resumes from results/lmax300_nside256_gibbs_real_double/checkpoint_chain_N.npz
# (which holds 1000 HMC samples) and collects 1000 more via PCG — IAT=1 at all l.
SAMPLER="gibbs"
ALM_SAMPLER="cg"
N_PCG_ITER=50
LMAX=300
NSIDE=256
SAMPLES=2000      # 1000 existing + 1000 new CG samples
BURNIN=0          # ignored on warm-start (resuming=True skips burnin)
DATA_MODE="real"
DATA_DIR="/cosma8/data/dp004/dc-hick2/Plank"
OUTPUT_DIR="$(pwd)/results/lmax${LMAX}_nside${NSIDE}_gibbs_${DATA_MODE}_double"

mkdir -p logs
mkdir -p $OUTPUT_DIR

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0
# Job-private TMPDIR: autograph writes transpiled sources to
# $TMPDIR/__autograph_generated_file*.py, which is node-local and not
# job-scoped by default. A stale entry from an earlier job on a reused
# node caused a bogus crash during debugging (2026-07-02) — see
# ROADMAP.md Phase 0b notes.
export TMPDIR="/tmp/diffcmb_cg_L300_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

echo "=== Phase 0b CG Sampler: L=${LMAX} NSIDE=${NSIDE} Chain ${SLURM_ARRAY_TASK_ID} ==="
echo "Python: $($PYTHON --version 2>&1)"
echo "Output dir: $OUTPUT_DIR"
free -h

$PYTHON -u scripts/run_sampler.py \
    --sampler $SAMPLER \
    --alm_sampler $ALM_SAMPLER \
    --n_pcg_iter $N_PCG_ITER \
    --lmax $LMAX \
    --nside $NSIDE \
    --n_samples $SAMPLES \
    --n_burnin $BURNIN \
    --data_mode $DATA_MODE \
    --data_dir $DATA_DIR \
    --map_steps 0 \
    --chain_id $SLURM_ARRAY_TASK_ID \
    --output_dir $OUTPUT_DIR \
    --double_precision
