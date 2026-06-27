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
#SBATCH --array=2,4

# Resubmit for synthetic chains 2 and 4 (OOM killed on first attempt)

SAMPLER="nuts"
LMAX=200
NSIDE=128
SAMPLES=5000
BURNIN=1000
STEP_SIZE=0.001
DATA_MODE="synthetic"
DATA_DIR="/cosma7/data/dp004/dc-hick2/Plank"
OUTPUT_DIR="$(pwd)/results/lmax${LMAX}_nside${NSIDE}_${SAMPLER}_${DATA_MODE}"

mkdir -p logs
mkdir -p $OUTPUT_DIR

module load python/3.9.19
source $(pwd)/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb

echo "===="
echo "Starting job ${SLURM_JOB_ID} at $(date) for user ${USER}."
echo "Running on nodes: $(hostname -s)"
echo "===="
echo "Starting job ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Working directory: $(pwd)"
echo "Python path: $(which python)"

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
