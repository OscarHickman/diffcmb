#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-23
#SBATCH -J ens_exact_l64
#SBATCH -o logs/ens_exact_l64_%A_%a.out
#SBATCH -e logs/ens_exact_l64_%A_%a.err
# 2026-09-23 production re-run (ROADMAP D3 + amplified-lensing validation).
# Exact geodesic lensing operator, corrected fiducial, amplified lensing A_phi,
# lmax=nside=64. One array task = one sky realization; the same realization
# index gives the SAME sky in every MODE, which is what figure 2 pairs on.
#   MODE=cl4    Block-4-ON, proper nu=6 prior  -> chain_rNNN.npz (figs 1,3,4, maps, convergence, fig 2 aware)
#   MODE=nocl4  Block-4-OFF                    -> chain_rNNN.npz in <dir>_nocl4 (3-block certification)
#   MODE=blind  lensing-blind fit, same data   -> blind_rNNN.npz (fig 2 blind)
# Submit: sbatch --export=ALL,MODE=cl4,AMP=...,NOISE=...,BURNIN=...,SAMPLES=... this.slurm
# Optional: NU (Block-4 prior nu, default 6.0) and TAG (suffix on the output
# directory, default none), e.g. TAG=_long for the T0.1c longer-chain re-run.
set -u
NU=${NU:-6.0}
TAG=${TAG:-}
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
BASE=results/analysis/ens_exact_l64_A${AMP}_n${NOISE}${TAG}
case $MODE in
  cl4)   OUT=$BASE;        EXTRA="--cl_phiphi_prior_nu $NU" ;;
  nocl4) OUT=${BASE}_nocl4; EXTRA="--no_sample_cl_phiphi" ;;
  blind) OUT=$BASE;        EXTRA="--cl_phiphi_prior_nu $NU --blind" ;;
  *) echo "unknown MODE=$MODE"; exit 2 ;;
esac
mkdir -p logs $OUT
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
# Tasks are packed several per node (no --exclusive), so each sees ONE of the
# node's four GPUs and TF must not reserve all its memory up front.
export CUDA_VISIBLE_DEVICES=$(( SLURM_ARRAY_TASK_ID % 4 ))
export TF_FORCE_GPU_ALLOW_GROWTH=true
echo "MODE=$MODE nu=$NU tag=${TAG} realization=${SLURM_ARRAY_TASK_ID} A_phi=$AMP noise=$NOISE burnin=$BURNIN samples=$SAMPLES -> $OUT"
.venv/bin/python -u scripts/coverage_ensemble_chain.py \
    --realization ${SLURM_ARRAY_TASK_ID} --lmax 64 --nside 64 \
    --n_burnin $BURNIN --n_samples $SAMPLES --n_lfs 10 --phi_n_lfs 240 \
    --phi_mass_matrix prior --noisesig $NOISE --phi_amplitude $AMP \
    --lensing_operator exact --fiducial corrected \
    --outdir $OUT --checkpoint_every 50 $EXTRA
echo "Realization ${SLURM_ARRAY_TASK_ID} exited with status $?"
