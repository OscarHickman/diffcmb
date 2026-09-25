#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska,bluefield1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -J pilot_exact_l64_A${AMP}
#SBATCH -o logs/pilot_exact_l64_%j.out
#SBATCH -e logs/pilot_exact_l64_%j.err
# Not dine2: results/analysis is a symlink into /cosma5, which dine2 does not mount
# (ROADMAP 2026-09-24). Memory sized from past MaxRSS instead of a whole node.
# 2026-09-23 (ROADMAP D3). Pilot for the exact-operator re-run: realization 0
# of the Block-4-ON nu=6 ensemble configuration, with --lensing_operator exact
# and --fiducial corrected, 100 post-burn-in sweeps, amplified lensing
# A_phi=${AMP}, pixel noise ${NOISE} (sbatch --export=AMP=...,NOISE=...,NSAMP=...).
# 2026-09-23: at noise 1 uK the phi|alm conditional is so sharp that Gibbs
# crawls (ESS ~4/100, phi power 6-8x truth); the QE S/N is cosmic-variance
# limited, so raising the noise to 10-30 uK/pix keeps the lensing information
# while loosening the alm-phi lock-in. Read: seconds/sweep (sets
# the ensemble walltime), alm/phi acceptance, phi power ratio to truth
# (phi_calibration_ok), and that the chain file records lensing_operator.
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
NOISE=${NOISE:-1.0}
NSAMP=${NSAMP:-100}
OUT=results/analysis/pilot_exact_lmax64_A${AMP}_n${NOISE}
mkdir -p logs $OUT
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
.venv/bin/python -u scripts/coverage_ensemble_chain.py \
    --realization 0 --lmax 64 --nside 64 \
    --n_burnin 100 --n_samples ${NSAMP} --n_lfs 10 --phi_n_lfs 240 --noisesig ${NOISE} \
    --phi_mass_matrix prior --cl_phiphi_prior_nu 6.0 \
    --lensing_operator exact --fiducial corrected --phi_amplitude ${AMP} \
    --outdir $OUT --checkpoint_every 50
rc=$?
echo "Exited with status $rc"
# Propagate the status so sacct shows FAILED rather than COMPLETED.
exit $rc
