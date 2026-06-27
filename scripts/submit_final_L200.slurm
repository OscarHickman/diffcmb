#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH -J cmb_final_L200
#SBATCH -o logs/cmb_L200_%j_%a.out
#SBATCH -e logs/cmb_L200_%j_%a.err
#SBATCH --array=1-4
#SBATCH --exclusive


# Final High-Resolution Real Planck Data Run (Mask-Cropped & High Memory)
SAMPLER="hmc"
LMAX=200
NSIDE=128
SAMPLES=5000
BURNIN=1000
STEP_SIZE=0.001
DATA_MODE="real"
DATA_DIR="/cosma8/data/dp004/dc-hick2/Plank"
OUTPUT_DIR="$(pwd)/results/lmax${LMAX}_nside${NSIDE}_${SAMPLER}_${DATA_MODE}"

mkdir -p logs
mkdir -p $OUTPUT_DIR

# Use absolute path for robustness
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

# Use the venv python directly
PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0

echo "Starting FINAL L=200 Real Run (L=${LMAX}, NSIDE=${NSIDE}) on GPU Partition (Optimized)"
echo "Chain ID: ${SLURM_ARRAY_TASK_ID}"
echo "Using Python: $($PYTHON --version 2>&1)"

# Check memory before starting
free -h

# Use -u for unbuffered output
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
    --output_dir $OUTPUT_DIR \
    --n_lfs 20

