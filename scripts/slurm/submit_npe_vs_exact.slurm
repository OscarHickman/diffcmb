#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH -J npe_vs_exact
#SBATCH -o logs/npe_vs_exact_%j.out
#SBATCH -e logs/npe_vs_exact_%j.err
# ROADMAP T1.1 (2026-10-07): the learned posterior (submit_npe_train_sbc.slurm)
# against the strict exact ensemble (ens_exact_l64_A3000_n30_nu30_strict,
# --cl_prior_nu 30 --cl_phiphi_prior_nu 30), sky by sky (figures 2 and 3).
# Submit with --dependency=afterany:<strict array>:<npe_train job>. Only skies
# whose chain file exists are compared; the count is printed, so a failed task
# shows as a missing sky rather than stopping the comparison.
# OUT overrides the output (default $FLOW/vs_exact_strict.npz, which make figure-npe reads).
# Not dine2: results/analysis is a symlink into /cosma5, which dine2 does not mount.
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
.venv/bin/python -u scripts/npe/npe_vs_exact.py --flow $FLOW --chains_dir $CHAINS \
    --realizations $REALS --n_learned 4000 --out ${OUT:-$FLOW/vs_exact_strict.npz}
rc=$?
echo "exit status $rc"
exit $rc
