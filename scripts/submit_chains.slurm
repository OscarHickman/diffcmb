#!/bin/bash
#SBATCH -A dp004
#SBATCH -p cosma7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
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
BURNIN=500
STEP_SIZE=0.01
OUTPUT_DIR="$(pwd)/results/lmax${LMAX}_nside${NSIDE}_${SAMPLER}"
# ---------------------

# Create necessary directories
mkdir -p logs
mkdir -p $OUTPUT_DIR

# Load module Python so the venv symlinks resolve on compute nodes
module load python/3.9.19

# Activate virtual environment
source $(pwd)/.venv/bin/activate

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

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
    --chain_id $SLURM_ARRAY_TASK_ID \
    --output_dir $OUTPUT_DIR
