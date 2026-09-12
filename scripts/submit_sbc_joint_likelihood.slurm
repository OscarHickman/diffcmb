#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH -J sbc_joint_lik
#SBATCH -o logs/sbc_joint_lik_%j.out
#SBATCH -e logs/sbc_joint_lik_%j.err

# 2026-09-12. SBC on the JOINT LIKELIHOOD as a test quantity, per Modrak et al.
# (arXiv:2211.02383) -- ROADMAP.md Impact strategy S3. The project currently
# ranks only parameter-derived quantities (per-l-bin field power, spectra);
# Modrak show data-dependent quantities, the joint likelihood above all, detect
# failures parameter ranks are blind to. This adds the recommended statistic to
# the headline Block-4-OFF ensemble (N=24, job 11955622 + 11965828).
#
# Run at TWO thinnings. The rank is unbiased under autocorrelation but gets
# noisier, and the project's standing practice is to show a statistic is
# thin-robust rather than quote one value (dashboard.md's headline row does the
# same). thin=10 gives ~60 draws/realization, thin=30 ~20.
#
# Runs on cosma5: dine2 is saturated by the bias-reduction replication and the
# Block-4-ON extension, and this is a few thousand lmax=64 lensing ops.

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
mkdir -p logs
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0

DIR=results/analysis/coverage_ensemble_lmax64_prior_nocl4_packingv2

for THIN in 10 30; do
  echo "################ thin=${THIN} ################"
  .venv/bin/python -u scripts/sbc_joint_likelihood.py \
      --indir $DIR --noisesig 1.0 --thin $THIN \
      --out $DIR/sbc_joint_likelihood_thin${THIN}.npz
done

echo "Exited with status $?"
