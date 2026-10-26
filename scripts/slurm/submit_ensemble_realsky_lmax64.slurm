#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska,cosma8-ska2,cosma7-shm
#SBATCH --exclude=mad01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --array=0-3
#SBATCH -J ens_realsky_l64
#SBATCH -o logs/ens_realsky_l64_%A_%a.out
#SBATCH -e logs/ens_realsky_l64_%A_%a.err
# ROADMAP T1.2a (2026-10-16): the strict ensemble's configuration on the REAL observing
# geometry: the Planck common temperature mask (data/planck_common_mask_int_ns64.fits,
# 75 % of the sky), a 5' beam + pixel window, and the ecliptic noise surrogate
# (noise_ecliptic_amp 0.5; Planck's per-pixel noise is not available). Truth is
# still drawn from the model's priors. Pilot first (validation-before-production):
# read the equilibration and strict-SBC numbers of these skies before scaling up.
# Output: results/analysis/ens_realsky_l64_A<AMP>_n<NOISE>_nu30_strict/
# Based on submit_ensemble_exact_lmax64.slurm (see it for MODE / env documentation).
# Exact geodesic lensing operator, corrected fiducial, amplified lensing A_phi,
# lmax=nside=64. One array task = one sky realization; the same realization
# index gives the SAME sky in every MODE, which is what figure 2 pairs on.
#   MODE=cl4    Block-4-ON, proper nu=6 prior  -> chain_rNNN.npz (figs 1,3,4, maps, convergence, fig 2 aware)
#   MODE=nocl4  Block-4-OFF                    -> chain_rNNN.npz in <dir>_nocl4 (3-block certification)
#   MODE=blind  lensing-blind fit, same data   -> blind_rNNN.npz (fig 2 blind)
#   MODE=phifixed  Blocks 1+2, phi held at phi_true (ROADMAP T0.1 a) -> chain_rNNN.npz;
#               keeps --cl_phiphi_prior_nu so phi_true is the SAME draw as the cl4 sky
#   MODE=truthstart  as cl4, but started at (a_lm, phi) = truth (ROADMAP T0.1 b,
#               two-start diagnostic, never SBC) -> chain_rNNN.npz
#   MODE=hotphi  as cl4, phi start x PHISCALE (default 4.5, power x 20): the hot-start
#               test of the A_phi = 1 real-data runaway (diagnostic, never SBC)
#               -> chain_rNNN.npz in <dir>_hotphi<PHISCALE>
# Submit: sbatch --export=ALL,MODE=cl4,AMP=...,NOISE=...,BURNIN=...,SAMPLES=... this.slurm
# Optional: NU (Block-4 prior nu, default 6.0) and TAG (suffix on the output
# directory, default none), e.g. TAG=_long for the T0.1c longer-chain re-run.
# CLNU (default none) adds --cl_prior_nu CLNU: the proper C_l^TT prior of strict
# SBC mode (ROADMAP T1.1 reference ensemble), with the truth C_l^TT drawn from it.
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
BASE=results/analysis/ens_realsky_l${LMAX}_A${AMP}_n${NOISE}${NSTAG}${TAG}
# GEOM overrides the observing geometry, e.g. GEOM="--beam_fwhm_arcmin 5 --noise_ecliptic_amp 0.5"
# with TAG=..._nomask: the same skies without the mask (the A_phi = 1 mask-edge test, 2026-10-20).
GEOM=${GEOM:---mask_file data/planck_common_mask_int_ns64.fits --beam_fwhm_arcmin 5 --noise_ecliptic_amp 0.5}
case $MODE in
  cl4)   OUT=$BASE;        EXTRA="--cl_phiphi_prior_nu $NU" ;;
  nocl4) OUT=${BASE}_nocl4; EXTRA="--no_sample_cl_phiphi" ;;
  blind) OUT=$BASE;        EXTRA="--cl_phiphi_prior_nu $NU --blind" ;;
  phifixed) OUT=$BASE;     EXTRA="--cl_phiphi_prior_nu $NU --phi_fixed_truth" ;;
  truthstart) OUT=$BASE;   EXTRA="--cl_phiphi_prior_nu $NU --start_truth" ;;
  hotphi) PHISCALE=${PHISCALE:-4.5}; OUT=${BASE}_hotphi${PHISCALE}
          EXTRA="--cl_phiphi_prior_nu $NU --phi_start_scale $PHISCALE" ;;
  *) echo "unknown MODE=$MODE"; exit 2 ;;
esac
EXTRA="$EXTRA --cl_init $CL_INIT"
# RESCALE=1 adds the NOT-CERTIFIED phi rescaling move (ROADMAP decision 1: the A_phi = 1
# trap fix; needs the user's sign-off) and writes to its own _rescale directory, so it can
# never be harvested with a certified ensemble.
if [ "${RESCALE:-0}" = 1 ]; then
  EXTRA="$EXTRA --phi_rescale_move --phi_rescale_proposal_scale ${RESCALE_SCALE:-0.1}"
  OUT=${OUT}_rescale
fi
CLNU=${CLNU:-}
[ -n "$CLNU" ] && EXTRA="$EXTRA --cl_prior_nu $CLNU"
mkdir -p logs $OUT
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/diffcmb
# Cap every pool at the allocation (AGENTS.md: pools size to the node, not SLURM).
NT=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT DUCC0_NUM_THREADS=$NT
export RAYON_NUM_THREADS=$NT NUMBA_NUM_THREADS=$NT
export TF_NUM_INTRAOP_THREADS=$NT TF_NUM_INTEROP_THREADS=2
export TF_ENABLE_ONEDNN_OPTS=0
# /cosma5-mounting partitions only (probe-verified 2026-09-07 and 2026-10-15; NOT
# dine2, and mad01 excluded: it lacks /cosma5; cosma7-shm adds mad02, which has
# it). Walltime: the pilot (12172310) took 9.5-9.6 h per sky on ga006 and the full-sky strict
# ensemble 12.3-12.4 h, so 16 h (chains checkpoint every 50 sweeps).
# results/analysis is a symlink into /cosma5, which dine2
# does not mount (every T0.1c task, jobs 12041916/12041917, died at its first
# checkpoint write for that reason). The chains are CPU-bound anyway: on dine2
# TF logged CUDA_ERROR_NO_DEVICE and never used the GPU.
echo "MODE=$MODE lmax=$LMAX nside=$NSIDE nu=$NU clnu=${CLNU:-none} tag=${TAG} realization=${SLURM_ARRAY_TASK_ID} A_phi=$AMP noise=$NOISE burnin=$BURNIN samples=$SAMPLES -> $OUT"
.venv/bin/python -u scripts/pipeline/coverage_ensemble_chain.py \
    --realization ${SLURM_ARRAY_TASK_ID} --lmax $LMAX --nside $NSIDE \
    --n_burnin $BURNIN --n_samples $SAMPLES --n_lfs 10 --phi_n_lfs 240 \
    --phi_mass_matrix prior --noisesig $NOISE --phi_amplitude $AMP \
    --lensing_operator exact --fiducial corrected \
    --outdir $OUT --checkpoint_every 50 $GEOM $EXTRA
rc=$?
echo "Realization ${SLURM_ARRAY_TASK_ID} exited with status $rc"
# Propagate the status so sacct shows FAILED rather than COMPLETED.
exit $rc
