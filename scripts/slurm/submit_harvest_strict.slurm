#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH -J harvest_strict
#SBATCH -o logs/harvest_strict_%j.out
#SBATCH -e logs/harvest_strict_%j.err
# ROADMAP T1.1 / T0.1 c (2026-10-07): harvest of the strict exact ensemble
# (job 12100331, ens_exact_l64_A3000_n30_nu30_strict: --cl_prior_nu 30
# --cl_phiphi_prior_nu 30, skies r000-r047) in one job:
#   1. refuse to run unless all 48 chains exist, hold 3600 samples and are strict;
#   2. strict SBC of both spectra against their drawn truths, full chain and
#      second half (strict_sbc_ranks.py), with the C_l collapse table (T0.1 c:
#      does the proper prior keep C_l off zero?);
#   3. field-level ranks (aggregate_coverage_ranks.py; strict here too) and the
#      joint-likelihood SBC (sbc_joint_likelihood.py).
# Submit with --dependency=afterany:12100331 (the guard catches failed tasks).
# Not dine2: results/analysis is a symlink into /cosma5, which dine2 does not mount.
set -u
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
mkdir -p logs
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/diffcmb
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 RAYON_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export DUCC0_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1
export TF_ENABLE_ONEDNN_OPTS=0
STRICT=results/analysis/ens_exact_l64_A3000_n30_nu30_strict

echo "================ 1. completeness guard ================"
.venv/bin/python - $STRICT <<'PY' || { echo "GUARD FAILED: not harvesting"; exit 1; }
import sys, numpy as np
bad = []
for r in range(48):
    f = f"{sys.argv[1]}/chain_r{r:03d}.npz"
    try:
        d = np.load(f)
    except OSError:
        bad.append(f"{f}: missing"); continue
    if d["alm_samples"].shape[0] != 3600:
        bad.append(f"{f}: {d['alm_samples'].shape[0]} samples")
    if not (np.isclose(float(d["cl_prior_nu"]), 30) and np.isclose(float(d["cl_phiphi_prior_nu"]), 30)):
        bad.append(f"{f}: not strict nu = 30")
print("\n".join(bad) if bad else "all 48 strict chains complete")
sys.exit(1 if bad else 0)
PY
grep -l Traceback logs/ens_exact_l64_12100331_*.err 2>/dev/null && echo "WARNING: a task logged a traceback (read it; a post-save phi-calibration raise keeps the chain)"

rc=0
run() { "$@" || { echo "FAILED ($?): $*"; rc=1; }; }
echo "================ 2. strict spectrum SBC + collapse table ================"
run .venv/bin/python -u scripts/strict_sbc_ranks.py --indir $STRICT --thin 20 \
    --out $STRICT/strict_sbc_ranks_thin20.npz
echo "---------------- second half ----------------"
run .venv/bin/python -u scripts/strict_sbc_ranks.py --indir $STRICT --thin 10 --burn_frac 0.5 \
    --out $STRICT/strict_sbc_ranks_thin10_half2.npz
echo "================ 3. field ranks and joint-likelihood SBC ================"
run .venv/bin/python -u scripts/aggregate_coverage_ranks.py --indir $STRICT --thin 10 \
    --out $STRICT/coverage_ranks_thin10.npz
run .venv/bin/python -u scripts/sbc_joint_likelihood.py --indir $STRICT --thin 40 \
    --out $STRICT/sbc_joint_likelihood_thin40.npz
echo "Exited with status $rc"
exit $rc
