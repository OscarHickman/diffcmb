#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska,cosma8-ska2,cosma7-shm
#SBATCH --exclude=mad01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH -J npe_c2st
#SBATCH -o logs/npe_c2st_%j.out
#SBATCH -e logs/npe_c2st_%j.err
# ROADMAP T1.1: classifier two-sample test, learned posterior (submit_npe_train_sbc.slurm)
# vs the strict exact ensemble, sky by sky (scripts/npe/npe_c2st.py). Only skies whose chain
# file exists are scored. Not dine2: results/analysis is a symlink into /cosma5.
set -u
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
A=results/analysis
FLOW=$A/npe_sims_l64_A3000_n30_nu30/flow_maf5x256
CHAINS=$A/ens_exact_l64_A3000_n30_nu30_strict
mkdir -p logs
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/diffcmb:$(pwd)/scripts
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 RAYON_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export DUCC0_NUM_THREADS=8 TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=1
export TF_ENABLE_ONEDNN_OPTS=0
[ -f $FLOW/sim_config.json ] || { echo "no trained flow at $FLOW"; exit 1; }
REALS=""
for r in $(seq 0 47); do
  [ -f $CHAINS/$(printf chain_r%03d.npz $r) ] && REALS="$REALS $r"
done
echo "skies with a chain: $(echo $REALS | wc -w)/48:$REALS"
[ -n "$REALS" ] || exit 1
.venv/bin/python -u scripts/npe/npe_c2st.py --flow $FLOW --chains_dir $CHAINS \
    --realizations $REALS --n_learned 4000 --out $FLOW/c2st_strict.npz
rc=$?
echo "exit status $rc"
exit $rc
