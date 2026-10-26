#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -J npe_train
#SBATCH -o logs/npe_train_%j.out
#SBATCH -e logs/npe_train_%j.err
# ROADMAP T1.1 (2026-10-07): train the spectra-level learned posterior on the
# 200 000 npe_simulate.py sims (submit_npe_simulate.slurm), then SBC it on 2000
# held-out sims with 199 draws each (figure 1 of the learned-vs-exact set).
# Submit with --dependency=afterok:<npe_sim array>.
# Not dine2: results/analysis is a symlink into /cosma5, which dine2 does not mount.
set -u
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
SIMS=results/analysis/npe_sims_l64_A3000_n30_nu30
FLOW=$SIMS/flow_maf5x256
mkdir -p logs
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/diffcmb:$(pwd)/scripts
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 RAYON_NUM_THREADS=16 NUMBA_NUM_THREADS=16
export DUCC0_NUM_THREADS=16 TF_NUM_INTRAOP_THREADS=16 TF_NUM_INTEROP_THREADS=1
export TF_ENABLE_ONEDNN_OPTS=0
echo "================ train ================"
.venv/bin/python -u scripts/npe/npe_train.py --sims_dir $SIMS --out $FLOW || exit 1
echo "================ SBC on held-out sims ================"
.venv/bin/python -u scripts/npe/npe_sbc.py --flow $FLOW --heldout $SIMS/heldout.npz \
    --n_sims 2000 --n_draws 199 --out $FLOW/sbc_heldout.npz
rc=$?
echo "exit status $rc"
exit $rc
