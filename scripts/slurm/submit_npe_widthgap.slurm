#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -J npe_gap
#SBATCH -o logs/npe_gap_%A_%a.out
#SBATCH -e logs/npe_gap_%A_%a.err
# ROADMAP T1.1 width gap (2026-10-14): the learned posterior is unbiased but
# 3-55 % wider than the exact one. One array task per variant; each trains,
# runs the held-out SBC (2000 sims x 199 draws) and the learned-vs-exact
# comparison on the 48 strict skies, exactly as the baseline flow_maf5x256.
#   route c, overfitting (v1 sims):  0 wd 1e-4 | 1 wd 1e-3 | 2 5x128 | 3 3x128 wd 1e-4
#   route a, richer summaries (v2):  4 wd 0    | 5 wd 1e-4
#   route b, learned compressor (v2, trains npe_compress.py first): 6
#   baseline (v1 sims, already trained by 12100400):  7
# Tasks 4-6 need the v2 sims (submit_npe_simulate.slurm with SUMMARY_SET=v2).
# SCORE_ONLY=1 (sbatch --export=ALL,SCORE_ONLY=1) skips training and the SBC and only
# reruns learned vs exact on an existing flow (2026-10-15: route c's first scoring
# ran on regressed v1 summaries; task 7 must reproduce 12100439 exactly).
# Not dine2: results/analysis is a symlink into /cosma5, which dine2 does not mount.
set -u
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
mkdir -p logs
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/diffcmb:$(pwd)/scripts
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 RAYON_NUM_THREADS=16 NUMBA_NUM_THREADS=16
export DUCC0_NUM_THREADS=16 TF_NUM_INTRAOP_THREADS=16 TF_NUM_INTEROP_THREADS=1
export TF_ENABLE_ONEDNN_OPTS=0
V1=results/analysis/npe_sims_l64_A3000_n30_nu30
V2=${V1}_v2
CHAINS=results/analysis/ens_exact_l64_A3000_n30_nu30_strict
COMP=""
case ${SLURM_ARRAY_TASK_ID} in
  0) SIMS=$V1; TAG=maf5x256_wd1e-4; ARGS="--weight_decay 1e-4" ;;
  1) SIMS=$V1; TAG=maf5x256_wd1e-3; ARGS="--weight_decay 1e-3" ;;
  2) SIMS=$V1; TAG=maf5x128;        ARGS="--hidden 128 128" ;;
  3) SIMS=$V1; TAG=maf3x128_wd1e-4; ARGS="--n_layers 3 --hidden 128 128 --weight_decay 1e-4" ;;
  4) SIMS=$V2; TAG=maf5x256;        ARGS="" ;;
  5) SIMS=$V2; TAG=maf5x256_wd1e-4; ARGS="--weight_decay 1e-4" ;;
  6) SIMS=$V2; TAG=maf5x256_comp;   ARGS=""; COMP=$V2/compressor ;;
  7) SIMS=$V1; TAG=maf5x256;        ARGS="" ;;
  *) echo "no variant ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac
FLOW=$SIMS/flow_$TAG
OUT=vs_exact_strict.npz
[ "${SLURM_ARRAY_TASK_ID}" = 7 ] && OUT=vs_exact_strict_repro.npz  # keep 12100439's file
echo "task ${SLURM_ARRAY_TASK_ID}: $FLOW ($ARGS${COMP:+, compressor $COMP})"
if [ "${SCORE_ONLY:-0}" = 1 ]; then
  echo "SCORE_ONLY: reusing the trained flow and its held-out SBC"
elif [ "${SLURM_ARRAY_TASK_ID}" = 7 ]; then
  echo "task 7 is the trained baseline: run it with SCORE_ONLY=1"; exit 1
else
if [ -n "$COMP" ]; then
  echo "================ train the map compressor ================"
  .venv/bin/python -u scripts/npe/npe_compress.py --sims_dir $SIMS --out $COMP || exit 1
  ARGS="$ARGS --compressor $COMP"
fi
echo "================ train ================"
.venv/bin/python -u scripts/npe/npe_train.py --sims_dir $SIMS --out $FLOW $ARGS || exit 1
echo "================ SBC on held-out sims ================"
.venv/bin/python -u scripts/npe/npe_sbc.py --flow $FLOW --heldout $SIMS/heldout.npz \
    --n_sims 2000 --n_draws 199 --out $FLOW/sbc_heldout.npz || exit 1
fi
echo "================ learned vs exact (strict ensemble) ================"
.venv/bin/python -u scripts/npe/npe_vs_exact.py --flow $FLOW --chains_dir $CHAINS \
    --realizations $(seq 0 47) --n_learned 4000 --out $FLOW/$OUT
rc=$?
echo "exit status $rc"
exit $rc
