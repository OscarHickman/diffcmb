#!/bin/bash
#SBATCH -A dp004
#SBATCH -p cosma7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -J cmb_real_lowl
#SBATCH -o logs/cmb_real_%j_%a.out
#SBATCH -e logs/cmb_real_%j_%a.err
#SBATCH --array=1-4

# Low-resolution run for real Planck data with Non-Centered Parameterization (NCP)
SAMPLER="nuts"
LMAX=64
NSIDE=32
SAMPLES=2000
BURNIN=500
STEP_SIZE=0.005
DATA_MODE="real"
DATA_DIR="/cosma7/data/dp004/dc-hick2/Plank"
OUTPUT_DIR="$(pwd)/results/lmax${LMAX}_nside${NSIDE}_${SAMPLER}_${DATA_MODE}"

mkdir -p logs
mkdir -p $OUTPUT_DIR

# Use absolute path for robustness
REPO_DIR="/cosma/apps/durham/dc-hick2/CMB_Advanced_Sampling"
cd $REPO_DIR

# Use the venv python directly
PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:.
export OMP_NUM_THREADS=8

echo "Starting Real Data Low-L run (L=${LMAX}, NSIDE=${NSIDE})"
echo "Chain ID: ${SLURM_ARRAY_TASK_ID}"
echo "Using Python: $($PYTHON --version 2>&1) at $(which $PYTHON)"

# Debug: check file content on node
grep "parameterization" scripts/run_sampler.py

# Use -u for unbuffered output to see progress in logs
$PYTHON -u scripts/run_sampler.py \
    --sampler $SAMPLER \
    --lmax $LMAX \
    --nside $NSIDE \
    --n_samples $SAMPLES \
    --n_burnin $BURNIN \
    --step_size $STEP_SIZE \
    --data_mode $DATA_MODE \
    --data_dir $DATA_DIR \
    --parameterization "non-centered" \
    --chain_id $SLURM_ARRAY_TASK_ID \
    --output_dir $OUTPUT_DIR
