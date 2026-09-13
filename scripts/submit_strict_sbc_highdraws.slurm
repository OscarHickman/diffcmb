#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=06:00:00
#SBATCH -J strict_hidraws
#SBATCH -o logs/strict_hidraws_%j.out
#SBATCH -e logs/strict_hidraws_%j.err

# 2026-09-13. The strict C_L^phiphi SBC rank and the Block-4 exactness PIT at
# ~60 draws/chain (thin=10), closing the one gap left by job 11986716: that job
# re-scored aggregate_coverage_ranks.py at both thinnings but NOT
# validate_coverage_rank_nulls.py, which is where the strict rank lives.
#
# This is the number that drove the "open sampling question": 0.4245 / KS_p
# 0.00395 at N=24 and ~8 draws/chain, already shown to soften to ~0.04 under a
# calibrated test. At 60 draws/chain the rank granularity is 10x finer and the
# KS test is properly calibrated there (FPR 5.7%/1.4% vs 30.3%/12.5% at 8
# draws), so this is the first properly-powered look at it.
#
# --control_lags 1,10,50: lag-1 alone is vacuous under the Block-4-ON funnel
# (phi lag-1 decorrelation +0.915), so a pass is only meaningful if a longer-lag
# control is rejected.

cd /cosma/apps/durham/dc-hick2/diffcmb
mkdir -p logs
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8 TF_ENABLE_ONEDNN_OPTS=0

for D in coverage_ensemble_lmax64_prior_cl4_properprior_packingv2 \
         coverage_ensemble_lmax64_prior_nocl4_packingv2; do
  for THIN in 10; do
    echo "################################################################"
    echo "### $D  --thin $THIN"
    echo "################################################################"
    .venv/bin/python -u scripts/validate_coverage_rank_nulls.py \
        --indir results/analysis/$D --thin $THIN --control_lags 1,10,50
  done
done
echo "Exited with status $?"
