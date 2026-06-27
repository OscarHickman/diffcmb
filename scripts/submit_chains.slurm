#!/bin/bash
#SBATCH -A dp004
#SBATCH -p cosma7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH -J cmb_sampling
#SBATCH -o logs/cmb_%j_%a.out
#SBATCH -e logs/cmb_%j_%a.err
#SBATCH --array=1-4

# --- Configuration ---
SAMPLER="nuts"
LMAX=200
NSIDE=128
SAMPLES=5000
BURNIN=1000
STEP_SIZE=0.001
# New options for data campaign
# Ensure DATA_MODE and DATA_DIR are correct for your current intent
DATA_MODE="real" # Can be 'synthetic' or 'real'
DATA_DIR="/cosma7/data/dp004/dc-hick2/Plank"
OUTPUT_DIR="$(pwd)/results/lmax${LMAX}_nside${NSIDE}_${SAMPLER}_${DATA_MODE}"
# ---------------------

# Create necessary directories
mkdir -p logs
mkdir -p $OUTPUT_DIR

# Load module Python so the venv symlinks resolve on compute nodes
module load python/3.9.19

# Activate virtual environment
source $(pwd)/.venv/bin/activate

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb

echo "Starting job ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Working directory: $(pwd)"
echo "Python path: $(which python)"

# Run the sampler
python scripts/run_sampler.py \
    --sampler $SAMPLER \
    --lmax $LMAX \
    --nside $NSIDE \
    --n_samples $SAMPLES \
    --n_burnin $BURNIN \
    --step_size $STEP_SIZE \
    --data_mode $DATA_MODE \
    --data_dir $DATA_DIR \
    --chain_id $SLURM_ARRAY_TASK_ID \
    --output_dir $OUTPUT_DIR
