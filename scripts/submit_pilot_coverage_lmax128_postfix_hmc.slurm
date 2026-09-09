#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH -J pilot_coverage_lmax128_postfix_hmc
#SBATCH -o logs/pilot_coverage_lmax128_postfix_hmc_%j.out
#SBATCH -e logs/pilot_coverage_lmax128_postfix_hmc_%j.err
#SBATCH --exclusive

# ===================================================================
# 2026-09-09 THE LENSING-AWARE lmax=128 CHAIN FOR THE C_l^TT BIAS-REDUCTION
# FIGURE (ROADMAP.md's differentiator-figures list), run for the FIRST TIME
# since the 2026-08-24 alm-ordering-fix and the 2026-09-06 dof restoration.
#
# WHY THIS MATTERS. Every prior lmax=128 phi-equilibration pilot on record
# (v2, phi240, extend, block/block_resume, mclmc/mclmc_v2 -- job numbers
# 116xxxxx-117xxxxx, git-dated 2026-07-31 to 2026-08-11, plus block_resume's
# underlying job 11795998) predates commit a16e9e6 (2026-08-24, alm-ordering
# fix), which achievements.md records as having "invalidated every phi-block
# result before this fix." achievements.md's "genuine, low-L-specific
# long-lived mode" verdict for lmax=128 (worst lag-1 0.996 in [2,10)) is
# therefore UNCONFIRMED post-fix, and has never been checked at all against
# the 2026-09-06 Im(a_{L,1}) dof restoration. This run does not re-litigate
# that verdict with a short diagnostic pilot first -- user directed launching
# the full chain directly, accepting that risk, rather than a cheap
# re-check. Read the .err/.out equilibration diagnostics (this script prints
# a GO/NO-GO verdict, same logic as the original gate) with that history in
# mind: a NO-GO here would be the FIRST post-fix data point on this
# question, not a re-confirmation of the old one.
#
# CONFIG. IDENTICAL sky/noise realization to
# results/analysis/lensing_blind_baseline_lmax128.npz (seed=0, lmax=128,
# nside=128, noisesig=1.0) so the recovered lensing-aware C_l^TT posterior
# is directly comparable to that lensing-blind reference -- this is the
# figure's whole point (ROADMAP.md: "the lensing-aware side still needs the
# equilibrated joint chain").
#
# --phi_sampler explicitly set to 'hmc'. pilot_coverage_equilibration.py
# DEFAULTS to 'mclmc' (promoted 2026-08-08), but MCLMC for the phi block is
# closed NO-GO (achievements.md: wins ESS/wall-clock-second everywhere but
# fails the stationarity gate twice, once getting WORSE after step-size
# tuning). phi_mass_matrix stays 'prior' (the only production-validated
# choice; 'block'/Nystrom closed, falsified at higher rank too). phi_n_lfs
# raised to 240 (the current lmax=64 production value, achievements.md's
# leapfrog-schedule entry) rather than the old lmax=128 pilots' 80.
# map_steps=2000/map_lr=0.01 are the validated production MAP-start
# settings, not run_gibbs_chain's weaker defaults.
#
# Block 4 is forced ON by phi_mass_matrix='prior' in this script (no
# independent toggle) with the default flat prior on C_L^phiphi -- fine
# here since the figure only needs the C_l^TT marginal, not a strict
# C_L^phiphi SBC rank (that caveat matters for calibration claims, not for
# this comparison).
#
# LENGTH. n_burnin=400 (MAP start already does most of the work),
# n_samples=1900 (2300 sweeps total) -- same order of magnitude as the
# 2026-08-06 "extend" pilot's 2500-sample target, chosen to give a stable
# C_l^TT posterior mean within the 3-day (72h) dine2 walltime cap even if
# per-sweep cost is on the high end of prior lmax=128 estimates
# (~30-60s/sweep pre-fix). checkpoint_every=50 sweeps: a walltime timeout
# loses at most 50 sweeps and this script's own checkpoint-resume path
# picks up cleanly (same pattern as every prior lmax=128 pilot).
#
# Job cap: 1 task. Re-check before submitting (limit is 200):
#   squeue -u dc-hick2 --noheader -r -o "%a" | grep -c durham
#
# HARVEST. Check .err for a traceback first (SLURM COMPLETED alone is not
# enough -- achievements.md's return-tuple-arity entry). Read the printed
# GO/NO-GO equilibration verdict (lag-1 autocorrelation + drift test) with
# the "first post-fix data point" framing above -- it is informative either
# way, not a pass/fail gate for whether to use the chain. Extract the
# posterior C_l^TT (log-Cl block of `alm_samples`, burn-in already excluded
# by run_gibbs_chain) and compare its mean against
# lensing_blind_baseline_lmax128.npz's cl_samples mean, bin by bin, for the
# bias-reduction figure. If walltime is hit before 2300 sweeps, resubmit
# against the same checkpoint rather than restarting.
# ===================================================================

LMAX=128
NSIDE=128
NOISESIG=1.0
BURNIN=400
SAMPLES=1900
N_LFS=10
PHI_N_LFS=240
MAP_STEPS=2000
MAP_LR=0.01
SEED=0

REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

mkdir -p logs results/analysis
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "=== pilot_coverage_lmax128_postfix_hmc job ${SLURM_JOB_ID} ==="
echo "Running on: $(hostname) at $(date)"
echo "git HEAD: $(git rev-parse --short HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"
$PYTHON --version
free -h

$PYTHON -u scripts/pilot_coverage_equilibration.py \
    --lmax $LMAX \
    --nside $NSIDE \
    --noisesig $NOISESIG \
    --n_burnin $BURNIN \
    --map_steps $MAP_STEPS \
    --map_lr $MAP_LR \
    --n_samples $SAMPLES \
    --n_lfs $N_LFS \
    --phi_sampler hmc \
    --phi_mass_matrix prior \
    --phi_n_lfs $PHI_N_LFS \
    --seed $SEED \
    --checkpoint_path results/analysis/pilot_coverage_lmax128_postfix_hmc_ckpt.npz \
    --checkpoint_every 50 \
    --out results/analysis/pilot_coverage_lmax128_postfix_hmc.npz

echo "Exited with status $?"
