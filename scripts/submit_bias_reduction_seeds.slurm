#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=02:00:00
#SBATCH -J bias_seeds
#SBATCH -o logs/bias_seeds_%j.out
#SBATCH -e logs/bias_seeds_%j.err

# 2026-09-13. Per-seed C_l^TT bias-reduction comparison + across-sky summary,
# taking the headline result from N=1 sky to N=4 (ROADMAP Next actions #1).
# Seed 0 is the original pair (job 11966631 aware / 11980570 blind); seeds 1-3
# are the replication (jobs 11984844 aware / 11984845 blind).
#
# Each seed's two chains MUST be the same sky: compare_cl_bias_reduction.py
# asserts their alm_true_packed agree before comparing, so a mismatched pair
# fails loudly rather than producing a plausible wrong figure.
#
# Skips any seed whose aware chain has not landed yet (seed 2 was still running
# at submission); the aggregator reports how many skies it actually found.

cd /cosma/apps/durham/dc-hick2/diffcmb
mkdir -p logs results/analysis/figures
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8 TF_ENABLE_ONEDNN_OPTS=0
P=".venv/bin/python"
F=results/analysis/figures

run_seed () {  # $1=seed  $2=blind npz  $3=aware npz
  if [ ! -f "$3" ]; then echo "### seed $1: aware chain not present yet, SKIPPED"; return; fi
  echo "############ seed $1 ############"
  $P -u scripts/compare_cl_bias_reduction.py \
      --blind "$2" --aware "$3" --lmax 128 \
      --exclude_unreliable --out $F/bias_seed$1
}

run_seed 0 results/analysis/lensing_blind_baseline_lmax128_packingv2.npz \
           results/analysis/pilot_coverage_lmax128_postfix_hmc.npz
for S in 1 2 3; do
  run_seed $S results/analysis/lensing_blind_baseline_lmax128_seed${S}.npz \
              results/analysis/bias_repl_aware_lmax128_seed${S}.npz
done

echo "############ ACROSS-SKY SUMMARY ############"
$P -u scripts/aggregate_bias_reduction_seeds.py \
    --inputs $F/bias_seed0.npz $F/bias_seed1.npz $F/bias_seed2.npz $F/bias_seed3.npz \
    --out $F/bias_reduction_seeds
echo "Exited with status $?"
