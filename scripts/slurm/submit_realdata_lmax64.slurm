#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska,cosma8-ska2,cosma7-shm,bluefield1
#SBATCH --exclude=mad01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --array=0-3
#SBATCH -J realdata_l64
#SBATCH -o logs/realdata_l64_%A_%a.out
#SBATCH -e logs/realdata_l64_%A_%a.err
# ROADMAP T1.2a, real data: Planck 2018 SMICA, band-limited in harmonic space by
# prepare_real_data.py (results/analysis/realdata_prep/smica_l64_ns64_b5.fits), on the
# T1.2a pilot's geometry (common mask, 5' beam + pixwin, ecliptic noise surrogate at
# sigma = NOISE, which the chain adds; SMICA's own noise is <= 0.26 muK in this band).
# Only the sky differs from the certified real-sky simulation (submit_ensemble_realsky).
# One sky, so every array task gets the SAME noise draw (--noise_realization NREAL) and a
# different start (--realization = task id): the tasks are chains for R-hat, not skies.
# Gate (ROADMAP): read harvest 12183262's strict SBC before submitting this.
# Submit: sbatch --export=ALL,AMP=<1|3000>,BURNIN=1000,SAMPLES=3600[,CLNU=][,MODE=blind] this.slurm
#   AMP   C_L^phiphi prior amplitude. 3000 is the certified configuration (a consistency
#         test the real sky, lensed at A_L = 1, should fail); 1 is physical (phi returns
#         its prior at lmax = 64). The user decides.
#   CLNU  proper nu prior on C_l^TT; default 30, the certified T1.2a configuration.
#         CLNU= (empty) gives the flat prior, which on real data does not recover from
#         the joint MAP start: find_map_estimate leaves ln C at its -12 clip for
#         l >= 60, and in smoke job 12184339 C_63 stayed there (D_63 ~ 0 against
#         Planck's ~1700 muK^2); the nu = 30 prior pulls it back in burn-in (pilot
#         12172310 started from the same clip).
#   MODE  aware (default) or blind (lensing-blind fit to the same data).
#   NREAL noise draw the chain adds (default 0); a second draw (NREAL=1, its own _nr1
#         directory) measures how far the posterior moves with the regularisation noise.
#   PHISCALE  phi start x this (default 1); != 1 is the hot-start diagnostic of the
#         A_phi = 1 chain-0 runaway, in its own _hotphi<PHISCALE> directory, never a posterior.
#   DATA  prepared map (default SMICA; realdata_prep/sevemR4_l64_ns64_b5.fits is PR4 SEVEM,
#         sevemR3_* its 2018 counterpart: the method systematic and the release check).
# Output: results/analysis/realdata_<map>_l64_A<AMP>_n<NOISE>[_clnu<CLNU>][_nr<NREAL>]/data_{chain,blind}_rNNN.npz
set -u
NOISE=${NOISE:-30}
NU=${NU:-30}
MODE=${MODE:-aware}
CLNU=${CLNU-30}
NREAL=${NREAL:-0}
PHISCALE=${PHISCALE:-1}
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
DATA=${DATA:-results/analysis/realdata_prep/smica_l64_ns64_b5.fits}
MAPTAG=$(basename $DATA | cut -d_ -f1)
OUT=results/analysis/realdata_${MAPTAG}_l64_A${AMP}_n${NOISE}${CLNU:+_clnu${CLNU}}$([ "$NREAL" != 0 ] && echo _nr$NREAL)$([ "$PHISCALE" != 1 ] && echo _hotphi$PHISCALE)${TAG:-}
GEOM="--mask_file data/planck_common_mask_int_ns64.fits --beam_fwhm_arcmin 5 --noise_ecliptic_amp 0.5"
EXTRA="--cl_phiphi_prior_nu $NU --data_map $DATA --noise_realization $NREAL --phi_start_scale $PHISCALE"
[ -n "$CLNU" ] && EXTRA="$EXTRA --cl_prior_nu $CLNU"
# RESCALE=1: the NOT-CERTIFIED phi rescaling move (ROADMAP decision 1), in its own _rescale directory.
if [ "${RESCALE:-0}" = 1 ]; then
  EXTRA="$EXTRA --phi_rescale_move --phi_rescale_proposal_scale ${RESCALE_SCALE:-0.1}"
  OUT=${OUT}_rescale
fi
case $MODE in
  aware) ;;
  blind) EXTRA="$EXTRA --blind" ;;
  *) echo "unknown MODE=$MODE"; exit 2 ;;
esac
mkdir -p logs $OUT
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/diffcmb
# Cap every pool at the allocation (AGENTS.md: pools size to the node, not SLURM).
NT=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT DUCC0_NUM_THREADS=$NT
export RAYON_NUM_THREADS=$NT NUMBA_NUM_THREADS=$NT
export TF_NUM_INTRAOP_THREADS=$NT TF_NUM_INTEROP_THREADS=2
export TF_ENABLE_ONEDNN_OPTS=0
# /cosma5-mounting partitions (results/analysis is on /cosma5); bluefield1 (32c, all
# three filesystems, probe 12171189) is overflow. Walltime as the real-sky pilot
# (9.5-12.4 h per chain); chains checkpoint every 50 sweeps.
echo "REAL DATA mode=$MODE A_phi=$AMP noise=$NOISE nu=$NU clnu=${CLNU:-none} nreal=$NREAL phiscale=$PHISCALE chain=${SLURM_ARRAY_TASK_ID} burnin=$BURNIN samples=$SAMPLES -> $OUT"
.venv/bin/python -u scripts/pipeline/coverage_ensemble_chain.py \
    --realization ${SLURM_ARRAY_TASK_ID} --lmax 64 --nside 64 \
    --n_burnin $BURNIN --n_samples $SAMPLES --n_lfs 10 --phi_n_lfs 240 \
    --phi_mass_matrix prior --noisesig $NOISE --phi_amplitude $AMP \
    --lensing_operator exact --fiducial corrected \
    --outdir $OUT --checkpoint_every 50 $GEOM $EXTRA
rc=$?
echo "Chain ${SLURM_ARRAY_TASK_ID} exited with status $rc"
exit $rc
