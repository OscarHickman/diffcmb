#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma8-shm,cosma8-ska,cosma8-shm2,cosma5,bluefield1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH -J null_alm_rank
#SBATCH -o logs/null_alm_rank_%j.out
#SBATCH -e logs/null_alm_rank_%j.err
# ROADMAP T0.1c: what an EXACT sampler gives for the a_lm power rank under the
# flat C_l prior with a fixed-spectrum truth (nominal and chain-calibrated
# effective noise), against the observed rank mean of both exact-operator
# ensembles. numpy/scipy only.
# Not dine2: results/analysis is a symlink into /cosma5, which dine2 does not mount.
set -u
# THIN must match fig1_validation.py --thin (Makefile FIG1_THIN); default 10
# writes the legacy null filename, any other value _thin<N>.
THIN=${THIN:-10}
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
mkdir -p logs
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/diffcmb
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
.venv/bin/python -u scripts/null_alm_power_rank_flat_prior.py \
    --indir results/analysis/ens_exact_l64_A3000_n30_nocl4 \
    --indir results/analysis/ens_exact_l64_A3000_n30_long \
    --indir results/analysis/ens_exact_l64_A3000_n30_nu30_long \
    --indir results/analysis/ens_exact_l64_A3000_n30 --n_null 2000 --thin $THIN
status=$?
echo "Exited with status $status"
exit $status
