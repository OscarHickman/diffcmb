#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=08:00:00
#SBATCH -J rescore_hidraws
#SBATCH -o logs/rescore_hidraws_%j.out
#SBATCH -e logs/rescore_hidraws_%j.err

# 2026-09-13. THE highest-value outstanding fix (ROADMAP Next actions #2 /
# "Rank-test remediation"): raise draws per chain from ~8 to ~60 by thinning by
# 10 instead of 90, and re-score both ensembles.
#
# WHY. The rank test was found miscalibrated at ~8 draws/chain: fed ranks from a
# PROVABLY CORRECT sampler, ks_uniform_p rejects at p<0.01 12.5% of the time on
# the pooled N=96 row, and the inflation GROWS with N. At 60 draws/chain it is
# well calibrated (5.7% / 1.4%). Power is also crippled at 8 draws -- a 0.3
# sigma posterior mean shift is caught 26.5% of the time. Both problems are
# cured by draws PER CHAIN, not by more chains. This is pure re-analysis of
# existing chains: no new sampling.
#
# THE TRADE this re-admits, and how it is monitored. thin=90 was chosen against
# a measured tau_int up to 42.5, so thin=10 leaves the draws autocorrelated.
# Autocorrelation does NOT bias the rank mean -- it inflates the rank SPREAD
# (U-shape) -- which is exactly what the new sd_u column measures (uniform =
# 0.2887). So read mean_u/cal_p for bias and sd_u for the autocorrelation cost,
# and compare against the thin=90 scores printed alongside. If sd_u balloons at
# thin=10 while mean_u holds, that is the expected autocorrelation signature,
# not a new defect.
#
# Runs BOTH thinnings so the comparison is in one log, on both ensembles, plus
# the joint-likelihood SBC at matched draw counts (at thin=30 the Block-4-ON
# value was 0.3771, the lowest number on the board, and deserves a properly
# powered look).

cd /cosma/apps/durham/dc-hick2/diffcmb
mkdir -p logs
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8 TF_ENABLE_ONEDNN_OPTS=0
P=".venv/bin/python"

for D in coverage_ensemble_lmax64_prior_nocl4_packingv2 \
         coverage_ensemble_lmax64_prior_cl4_properprior_packingv2; do
  for THIN in 90 10; do
    echo "################################################################"
    echo "### $D   --thin $THIN   (draws/chain ~ $((600/THIN)))"
    echo "################################################################"
    $P -u scripts/aggregate_coverage_ranks.py \
        --indir results/analysis/$D --thin $THIN
  done
  echo "### joint-likelihood SBC, $D, thin=5 (~120 draws/chain)"
  $P -u scripts/sbc_joint_likelihood.py \
      --indir results/analysis/$D --noisesig 1.0 --thin 5 \
      --out results/analysis/$D/sbc_joint_likelihood_thin5.npz
done
echo "Exited with status $?"
