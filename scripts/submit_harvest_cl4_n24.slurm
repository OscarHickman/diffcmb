#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH -J harvest_cl4_n24
#SBATCH -o logs/harvest_cl4_n24_%j.out
#SBATCH -e logs/harvest_cl4_n24_%j.err

# 2026-09-12. Harvest the Block-4-ON proper-prior ensemble at N=24 (job
# 11965813 realizations 0-11 + job 11980637 realizations 12-23). This is the
# decision-critical run for ROADMAP Impact strategy S2: does the strict
# C_L^phiphi SBC residual, localised to [10,30) and measured at
# 0.4245/KS_p=0.00395 with the [10,30) bin at 0.2812/KS_p=0.0047 at N=12,
# RELAX (small-N, as the Block-4-OFF [30,60) flag did) or SHARPEN (a real,
# localised defect) at N=24?
#
# Also runs the new joint-likelihood SBC (Modrak et al.) on this ensemble --
# the more interesting of the two configurations for that statistic, since a
# data-dependent test quantity is the one most likely to say something the
# phi-power ranks cannot about the open residual.
#
# --control_lags 1,10,50 because lag-1 alone is vacuous here (phi lag-1
# decorrelation +0.915 under the Block-4-ON funnel).

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
mkdir -p logs
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0

DIR=results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_packingv2

echo "############ A) ranks, N=24 ############"
.venv/bin/python -u scripts/aggregate_coverage_ranks.py --indir $DIR --thin 90

echo "############ B) nulls + STRICT C_L^phiphi SBC, N=24 ############"
.venv/bin/python -u scripts/validate_coverage_rank_nulls.py --indir $DIR \
    --thin 90 --control_lags 1,10,50

echo "############ C) joint-likelihood SBC (Modrak) ############"
for THIN in 10 30; do
  echo "---- thin=${THIN} ----"
  .venv/bin/python -u scripts/sbc_joint_likelihood.py \
      --indir $DIR --noisesig 1.0 --thin $THIN \
      --out $DIR/sbc_joint_likelihood_thin${THIN}.npz
done

echo "Exited with status $?"
