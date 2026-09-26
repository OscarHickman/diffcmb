#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma8-shm,cosma8-ska,cosma8-shm2,cosma8-shm3,cosma5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH -J harvest_t01e
#SBATCH -o logs/harvest_t01e_%j.out
#SBATCH -e logs/harvest_t01e_%j.err
# ROADMAP T0.1e: N = 48 harvest of the nu = 30 long ensemble once the fresh skies
# r024-r047 (Block-4-ON job 12065119, blind pair 12065120) are in. dashboard.md
# "CURRENT (2026-09-26)" checklist steps 1-4 in one job:
#   1. refuse to run unless all 48 fresh chain/blind files exist and are complete;
#   2. keep the N = 24 outputs as *_n24.npz, then redo the thin-50 a_lm null,
#      joint-likelihood SBC and rank nulls at N = 48;
#   3. expected_aware_bias_flat_prior (before the figures: figure 2 draws its
#      band only from a result made on the same 48 skies);
#   4. make figures;  5. inspect_fig4_cells.
# Submit with --dependency=afterany:12065119 (afterok would never fire if one task
# failed; the guard below catches missing chains instead).
# Not dine2: results/analysis is a symlink into /cosma5, which dine2 does not mount.
set -u
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR
mkdir -p logs
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/diffcmb
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 RAYON_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1
export TF_ENABLE_ONEDNN_OPTS=0
NU30=results/analysis/ens_exact_l64_A3000_n30_nu30_long

echo "================ 1. completeness guard ================"
missing=0
for r in $(seq 24 47); do
  for kind in chain blind; do
    f=$(printf "%s/%s_r%03d.npz" $NU30 $kind $r)
    [ -s "$f" ] || { echo "MISSING $f"; missing=1; }
  done
done
# A task that died at a checkpoint leaves a short chain: every Block-4-ON chain
# must hold the full 3600 samples.
.venv/bin/python - $NU30 <<'EOF' || missing=1
import sys, glob, numpy as np
bad = []
for f in sorted(glob.glob(f"{sys.argv[1]}/chain_r0[2-4][0-9].npz")):
    n = np.load(f)["alm_samples"].shape[0]
    if n != 3600:
        bad.append((f, n))
for f, n in bad:
    print(f"SHORT {f}: {n} samples")
sys.exit(1 if bad else 0)
EOF
grep -l Traceback logs/ens_exact_l64_12065119_*.err 2>/dev/null && missing=1
if [ $missing -ne 0 ]; then
  echo "Harvest refused: resubmit the missing realizations (dashboard.md checklist step 1)."
  exit 2
fi
echo "48 chain+blind files present, all chains 3600 samples"

rc=0
run() { "$@" || { echo "FAILED ($?): $*"; rc=1; }; }
for f in null_alm_power_rank_flat_prior_thin50 sbc_joint_likelihood_thin40 \
         sbc_joint_likelihood_thin10_half2 expected_aware_bias_flat_prior; do
  [ -e $NU30/$f.npz ] && [ ! -e $NU30/${f}_n24.npz ] && cp -p $NU30/$f.npz $NU30/${f}_n24.npz
done

echo "================ 2. a_lm null (thin 50), SBC, rank nulls at N = 48 ================"
run .venv/bin/python -u scripts/null_alm_power_rank_flat_prior.py \
    --indir $NU30 --n_null 2000 --thin 50
run .venv/bin/python -u scripts/sbc_joint_likelihood.py --indir $NU30 --thin 40 \
    --out $NU30/sbc_joint_likelihood_thin40.npz
run .venv/bin/python -u scripts/sbc_joint_likelihood.py --indir $NU30 --thin 10 \
    --burn_frac 0.5 --out $NU30/sbc_joint_likelihood_thin10_half2.npz
run .venv/bin/python -u scripts/validate_coverage_rank_nulls.py --indir $NU30 --thin 10

echo "================ 3. expected aware bias (figure 2 band) ================"
run .venv/bin/python -u scripts/expected_aware_bias_flat_prior.py --indir $NU30

echo "================ 4. make figures ================"
run make figures

echo "================ 5. figure 4 cells ================"
run .venv/bin/python -u scripts/inspect_fig4_cells.py --indir $NU30
echo "Exited with status $rc"
exit $rc
