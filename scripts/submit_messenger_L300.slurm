#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH -J cmb_msgr_L300
#SBATCH -o logs/cmb_msgr_L300_%j_%a.out
#SBATCH -e logs/cmb_msgr_L300_%j_%a.err
#SBATCH --array=1-4
#SBATCH --exclusive

# Phase 0c Step 6: production validation chain for the messenger-field
# alm|C_l sampler (block-diagonal-by-m A^T A correction, m_group_size=5 --
# chosen per ROADMAP.md: matches the exact-dense correction's accuracy at
# validation scale, ~354.68s/sweep at lmax=300 per benchmark job 11572610).
# Fresh chains (not warm-started from the CG/HMC checkpoint dir, whose
# alm_state provenance is ambiguous after several superseded CG attempts --
# see ROADMAP.md Phase 0b/0c). Goal: verify ESS ~= N at all multipoles and
# C_l agreement with the Phase 0 HMC posterior.
SAMPLER="gibbs"
ALM_SAMPLER="messenger"
N_MESSENGER_ITER=100
LMAX=300
NSIDE=256
SAMPLES=520       # n_burnin=20 + 500 production samples
BURNIN=20
DATA_MODE="real"
DATA_DIR="/cosma8/data/dp004/dc-hick2/Plank"
OUTPUT_DIR="$(pwd)/results/lmax${LMAX}_nside${NSIDE}_gibbs_${DATA_MODE}_messenger"

mkdir -p logs
mkdir -p $OUTPUT_DIR

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
# Matrix-free ducc0 SHT is CPU-only -- no GPU needed for the messenger path.
export CUDA_VISIBLE_DEVICES=""
# Job-private TMPDIR: autograph writes transpiled sources to
# $TMPDIR/__autograph_generated_file*.py, node-local and not job-scoped by
# default -- see ROADMAP.md Phase 0b notes on the stale-cache crash this
# caused previously.
export TMPDIR="/tmp/diffcmb_msgr_L300_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

echo "=== Phase 0c Step 6 Messenger Sampler: L=${LMAX} NSIDE=${NSIDE} Chain ${SLURM_ARRAY_TASK_ID} ==="
echo "Python: $($PYTHON --version 2>&1)"
echo "Output dir: $OUTPUT_DIR"
free -h

$PYTHON -u scripts/run_sampler.py \
    --sampler $SAMPLER \
    --alm_sampler $ALM_SAMPLER \
    --n_messenger_iter $N_MESSENGER_ITER \
    --messenger_use_block_correction \
    --messenger_m_group_size 5 \
    --use_matrixfree_sht \
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
