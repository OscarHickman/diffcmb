#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=36:00:00
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
# CL_INIT (default map = production) selects the ln C_l start; CL_INIT=data is the
# ROADMAP T0.1 pilot (needs sign-off before production use).
# LMAX (default 64) sets lmax = nside and the l<LMAX> in the directory name;
# used by ROADMAP T0.1 (band-limit test at a smaller lmax = nside).
# NSIDE (default LMAX) decouples the pixel scale from the band limit; when it
# differs from LMAX the directory gets an _ns<NSIDE> suffix.
set -u
NU=${NU:-6.0}
CL_INIT=${CL_INIT:-map}
TAG=${TAG:-}
LMAX=${LMAX:-64}
NSIDE=${NSIDE:-$LMAX}
NSTAG=""; [ "$NSIDE" != "$LMAX" ] && NSTAG="_ns${NSIDE}"
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
BASE=results/analysis/ens_exact_l${LMAX}_A${AMP}_n${NOISE}${NSTAG}${TAG}
case $MODE in
  cl4)   OUT=$BASE;        EXTRA="--cl_phiphi_prior_nu $NU" ;;
  nocl4) OUT=${BASE}_nocl4; EXTRA="--no_sample_cl_phiphi" ;;
  blind) OUT=$BASE;        EXTRA="--cl_phiphi_prior_nu $NU --blind" ;;
  *) echo "unknown MODE=$MODE"; exit 2 ;;
esac
EXTRA="$EXTRA --cl_init $CL_INIT"
mkdir -p logs $OUT
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
# Cap every pool at the allocation (AGENTS.md: pools size to the node, not SLURM).
export OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 DUCC0_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2
export TF_ENABLE_ONEDNN_OPTS=0
# /cosma5-mounting partitions only (probe-verified 2026-09-07; NOT dine2, and
# not cosma7-shm/ska2, whose mad01 lacks /cosma5): results/analysis is a
# symlink into /cosma5, which dine2
# does not mount (every T0.1c task, jobs 12041916/12041917, died at its first
# checkpoint write for that reason). The chains are CPU-bound anyway: on dine2
# TF logged CUDA_ERROR_NO_DEVICE and never used the GPU.
echo "MODE=$MODE lmax=$LMAX nside=$NSIDE nu=$NU tag=${TAG} realization=${SLURM_ARRAY_TASK_ID} A_phi=$AMP noise=$NOISE burnin=$BURNIN samples=$SAMPLES -> $OUT"
.venv/bin/python -u scripts/coverage_ensemble_chain.py \
    --realization ${SLURM_ARRAY_TASK_ID} --lmax $LMAX --nside $NSIDE \
    --n_burnin $BURNIN --n_samples $SAMPLES --n_lfs 10 --phi_n_lfs 240 \
    --phi_mass_matrix prior --noisesig $NOISE --phi_amplitude $AMP \
    --lensing_operator exact --fiducial corrected \
    --outdir $OUT --checkpoint_every 50 $EXTRA
rc=$?
echo "Realization ${SLURM_ARRAY_TASK_ID} exited with status $rc"
# Propagate the status so sacct shows FAILED rather than COMPLETED.
exit $rc
