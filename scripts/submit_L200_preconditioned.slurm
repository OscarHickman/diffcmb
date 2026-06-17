#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH -J cmb_L200_prec
#SBATCH -o logs/cmb_L200_prec_%j_%a.out
#SBATCH -e logs/cmb_L200_prec_%j_%a.err
#SBATCH --array=1-4
#SBATCH --exclusive

# L=200 run with diagonal mass matrix preconditioning.
# Centered parameterization so the mass matrix (1/sqrt(Cl)) correctly
# rescales alm parameters to unit posterior variance.
# Step size and n_lfs are chosen for the preconditioned (near-unit) space.
SAMPLER="hmc"
LMAX=200
NSIDE=128
SAMPLES=5000
BURNIN=2000
STEP_SIZE=0.1
DATA_MODE="real"
DATA_DIR="/cosma8/data/dp004/dc-hick2/Plank"
OUTPUT_DIR="$(pwd)/results/lmax${LMAX}_nside${NSIDE}_${SAMPLER}_${DATA_MODE}_preconditioned"

mkdir -p logs
mkdir -p $OUTPUT_DIR

REPO_DIR="/cosma/apps/durham/dc-hick2/CMB_Advanced_Sampling"
cd $REPO_DIR

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:.
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0

echo "Starting PRECONDITIONED L=200 Real Run (L=${LMAX}, NSIDE=${NSIDE})"
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
    --data_mode $DATA_MODE \
    --data_dir $DATA_DIR \
    --parameterization "centered" \
    --n_lfs 10 \
    --map_steps 500 \
    --chain_id $SLURM_ARRAY_TASK_ID \
    --output_dir $OUTPUT_DIR
