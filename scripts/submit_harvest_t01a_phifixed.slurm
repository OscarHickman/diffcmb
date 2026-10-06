#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH -J harvest_t01a
#SBATCH -o logs/harvest_t01a_%j.out
#SBATCH -e logs/harvest_t01a_%j.err
# ROADMAP T0.1 a (2026-10-07): harvest of the phi-fixed run (job 12099339,
# MODE=phifixed, skies r032-r039) in one job:
#   1. refuse to run unless all 8 chains exist, hold 3600 samples and record
#      phi_fixed_truth = True;
#   2. diagnose_alm_band_edge on the phi-fixed run (ranks, C by quarter, sd(z),
#      shrinkage slope), with the control beside it for the same skies;
#   3. compare_band_edge_paired: phi-fixed and control paired by sky against
#      the exact dense reference, without r036 (collapsed sky) and with it.
# Reading: C_63 back at the dense value -> the a_lm/C_l blocks are exact and the
# deficit comes from marginalising phi; still ~0.935 -> the a_lm block or the
# likelihood at l = lmax-1.
# Submit with --dependency=afterany:12099339 (the guard catches failed tasks).
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
A=results/analysis
PHIFIXED=$A/ens_exact_l64_A3000_n30_nu30_phifixed
CONTROL=$A/ens_exact_l64_A3000_n30_nu30_long_r032_r039
DENSE=$A/exact_dense_alm_reference_l64_nu30

echo "================ 1. completeness guard ================"
.venv/bin/python - $PHIFIXED <<'EOF' || { echo "GUARD FAILED: not harvesting"; exit 1; }
import sys, numpy as np
bad = []
for r in range(32, 40):
    f = f"{sys.argv[1]}/chain_r{r:03d}.npz"
    try:
        d = np.load(f)
    except OSError:
        bad.append(f"{f}: missing"); continue
    if d["alm_samples"].shape[0] != 3600:
        bad.append(f"{f}: {d['alm_samples'].shape[0]} samples")
    if not bool(d["phi_fixed_truth"]):
        bad.append(f"{f}: phi_fixed_truth is False")
    if not np.allclose(d["phi_samples"][-1], d["phi_true_packed"], rtol=1e-12, atol=0):
        bad.append(f"{f}: phi moved")
print("\n".join(bad) if bad else "all 8 phi-fixed chains complete")
sys.exit(1 if bad else 0)
EOF

echo "================ 2. band-edge diagnostic ================"
.venv/bin/python -u scripts/diagnose_alm_band_edge.py --indir $PHIFIXED --indir $CONTROL --split 0
rc=$?

echo "================ 3. paired vs the dense reference (r036 excluded) ================"
.venv/bin/python -u scripts/compare_band_edge_paired.py --ref $DENSE \
    --indir $CONTROL --indir $PHIFIXED --exclude 36 --out $PHIFIXED/compare_band_edge_paired.npz || rc=1
echo "================ 3b. the same with r036 ================"
.venv/bin/python -u scripts/compare_band_edge_paired.py --ref $DENSE \
    --indir $CONTROL --indir $PHIFIXED || rc=1
echo "harvest exit status $rc"
exit $rc
