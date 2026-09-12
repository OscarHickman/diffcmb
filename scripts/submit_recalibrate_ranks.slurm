#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=04:00:00
#SBATCH -J recal_ranks
#SBATCH -o logs/recal_ranks_%j.out
#SBATCH -e logs/recal_ranks_%j.err

# 2026-09-12. Re-read BOTH production ensembles through the corrected rank test
# (aggregate_coverage_ranks.discrete_uniform_p + rank_spread). Every FLAG and
# every "pass" on record was scored with ks_uniform_p, which compares discrete
# ranks to a continuous uniform and rejects a PROVABLY CORRECT sampler 12.5% of
# the time at p<0.01 on the pooled N=96/8-draw row. This reprints both
# ensembles with the calibrated p-value and the sd_u under-dispersion
# diagnostic, so the claims can be restated against a valid test.

cd /cosma/apps/durham/dc-hick2/diffcmb
mkdir -p logs
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8 TF_ENABLE_ONEDNN_OPTS=0

for D in coverage_ensemble_lmax64_prior_nocl4_packingv2 \
         coverage_ensemble_lmax64_prior_cl4_properprior_packingv2; do
  echo "################################################################"
  echo "### $D"
  echo "################################################################"
  .venv/bin/python -u scripts/aggregate_coverage_ranks.py \
      --indir results/analysis/$D --thin 90
done
echo "Exited with status $?"
