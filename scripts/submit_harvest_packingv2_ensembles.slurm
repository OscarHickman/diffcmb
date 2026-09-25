#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska,bluefield1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -J harvest_packingv2
#SBATCH -o logs/harvest_packingv2_%j.out
#SBATCH -e logs/harvest_packingv2_%j.err
# Not dine2: results/analysis is a symlink into /cosma5, which dine2 does not mount
# (ROADMAP 2026-09-24). Memory sized from past MaxRSS instead of a whole node.

# ===================================================================
# 2026-09-11 HARVEST of the two completed packing-v2 coverage ensembles
# (ROADMAP.md "Next actions" #1 and #2). Analysis only -- reads the saved
# chains, writes no new chain state. Run under SLURM rather than on the
# login node (CLAUDE.md: no heavy python on a login node); the aggregator
# reduces ~1 GB of chain_r*.npz per ensemble.
#
#  A) job 11965828 extension: nocl4 packing-v2 ensemble now N=24. Re-run the
#     aggregator over ALL 24 realizations and compare the phi [30,60) bin's
#     KS_p against the N=12 value (0.005), and the pooled headline pair
#     against (phi 0.4792, alm 0.5130).
#  B) job 11965813: Block-4-ON proper-prior (nu=6) under the restored 2L+1
#     packing. Decisive number is the STRICT C_L^phiphi SBC rank vs job
#     11903182's pre-restoration 0.3802 / KS_p=0.0013.
#
# --thin 90 is the starting value carried from job 11955622; the aggregator
# prints this run's own tau_int, so re-derive and re-run if it disagrees
# (harvest checklist, dashboard.md). --control_lags 1,10,50 because lag-1
# alone makes the Block 4 PIT vacuous under the Block-4-ON funnel.
# ===================================================================

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
mkdir -p logs results/analysis
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0

NOCL4=results/analysis/coverage_ensemble_lmax64_prior_nocl4_packingv2
CL4=results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_packingv2

echo "############ A) nocl4 packing-v2, N=24 (job 11965828 extension) ############"
$PYTHON -u scripts/aggregate_coverage_ranks.py --indir $NOCL4 --thin 90
echo "############ A2) nulls for nocl4 ############"
$PYTHON -u scripts/validate_coverage_rank_nulls.py --indir $NOCL4 --thin 90 --control_lags 1,10,50

echo "############ B) Block-4-ON proper prior, restored packing (job 11965813) ############"
$PYTHON -u scripts/aggregate_coverage_ranks.py --indir $CL4 --thin 90
echo "############ B2) nulls + STRICT C_L^phiphi SBC for cl4 ############"
$PYTHON -u scripts/validate_coverage_rank_nulls.py --indir $CL4 --thin 90 --control_lags 1,10,50

rc=$?
echo "Exited with status $rc"
# Propagate the status so sacct shows FAILED rather than COMPLETED.
exit $rc
