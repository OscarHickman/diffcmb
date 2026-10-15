#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -J npe_lc
#SBATCH -o logs/npe_lc_%A_%a.out
#SBATCH -e logs/npe_lc_%A_%a.err
# ROADMAP T1.1 learning curve (2026-10-15): no width-gap route (regularisation,
# capacity, richer summaries, compressor) moved the ratios. Train the baseline
# flow on 50k and 100k of the 200k v1 sims; a ratio that does not fall with
# more data points to the summaries' information cost, not a training deficit.
#   0: 50k (train_0)   1: 100k (train_0,1)    baseline 200k = flow_maf5x256
set -u
cd /cosma/apps/durham/dc-hick2/diffcmb
mkdir -p logs
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/diffcmb:$(pwd)/scripts
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK} RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK} NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export DUCC0_NUM_THREADS=${SLURM_CPUS_PER_TASK} TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK} TF_NUM_INTEROP_THREADS=1 TF_ENABLE_ONEDNN_OPTS=0
BASE=results/analysis/npe_sims_l64_A3000_n30_nu30
CHAINS=results/analysis/ens_exact_l64_A3000_n30_nu30_strict
case ${SLURM_ARRAY_TASK_ID} in
  0) SIMS=${BASE}_n50k ;;
  1) SIMS=${BASE}_n100k ;;
  *) echo "no variant"; exit 1 ;;
esac
FLOW=$SIMS/flow_maf5x256
.venv/bin/python -u scripts/npe_train.py --sims_dir $SIMS --out $FLOW || exit 1
.venv/bin/python -u scripts/npe_sbc.py --flow $FLOW --heldout $SIMS/heldout.npz \
    --n_sims 2000 --n_draws 199 --out $FLOW/sbc_heldout.npz || exit 1
.venv/bin/python -u scripts/npe_vs_exact.py --flow $FLOW --chains_dir $CHAINS \
    --realizations $(seq 0 47) --n_learned 4000 --out $FLOW/vs_exact_strict.npz
echo "exit status $?"
