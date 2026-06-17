#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=1500G
#SBATCH --time=72:00:00
#SBATCH -J cmb_gibbs_L300_dp
#SBATCH -o logs/cmb_gibbs_L300_dp_%j_%a.out
#SBATCH -e logs/cmb_gibbs_L300_dp_%j_%a.err
#SBATCH --array=1-4
#SBATCH --exclusive

# Gibbs sampler: exact C_l | alm (inverse-Gamma) + HMC alm | C_l in double precision.
# Resolution: L=300, captures the trough between peaks 1 and 2.
# Precision: double precision (complex128/float64) for matrix operations to prevent gradient noise collapse.
SAMPLER="gibbs"
LMAX=300
NSIDE=256
SAMPLES=2000
BURNIN=500
STEP_SIZE=0.01
N_LFS=20
DATA_MODE="real"
DATA_DIR="/cosma8/data/dp004/dc-hick2/Plank"
OUTPUT_DIR="$(pwd)/results/lmax${LMAX}_nside${NSIDE}_${SAMPLER}_${DATA_MODE}_double"

mkdir -p logs
mkdir -p $OUTPUT_DIR

REPO_DIR="/cosma/apps/durham/dc-hick2/CMB_Advanced_Sampling"
cd $REPO_DIR

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:.
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "Starting Gibbs L=300 Real Double Precision Run (L=${LMAX}, NSIDE=${NSIDE})"
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
    --chain_id $SLURM_ARRAY_TASK_ID \
    --output_dir $OUTPUT_DIR \
    --double_precision
