#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH -J harvest_t01b
#SBATCH -o logs/harvest_t01b_%j.out
#SBATCH -e logs/harvest_t01b_%j.err
# ROADMAP T0.1 b (2026-10-07): harvest of the two-start test (MODE=truthstart,
# skies r032-r039, production configuration started at (a_lm, phi) = truth) in
# one job:
#   1. refuse to run unless all 8 chains exist, hold 3600 samples, record
#      start_truth = True and really started phi and a_lm at the truth;
#   2. diagnose_alm_band_edge on the truth-start run (ranks, C by quarter, sd(z),
#      shrinkage slope), with the MAP-start control beside it for the same skies;
#   3. compare_band_edge_paired: truth start and MAP start paired by sky against
#      the exact dense (phi-known) reference, without r036 (collapsed sky) and with it.
# Reading: truth-start C_63 drifts down to ~0.935 and stays -> the deficit is the
# stationary phi-marginal posterior (figure 1 needs a phi-marginal reference);
# truth start stays near 1.0 while the MAP start stays near 0.935 -> a-phi mixing
# at the band edge is too slow for 3600 sweeps.
# Submit with --dependency=afterany:<truthstart job> (the guard catches failed tasks).
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
TRUTHSTART=$A/ens_exact_l64_A3000_n30_nu30_truthstart
CONTROL=$A/ens_exact_l64_A3000_n30_nu30_long_r032_r039
DENSE=$A/exact_dense_alm_reference_l64_nu30

echo "================ 1. completeness guard ================"
.venv/bin/python - $TRUTHSTART <<'EOF' || { echo "GUARD FAILED: not harvesting"; exit 1; }
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
    if not bool(d["start_truth"]):
        bad.append(f"{f}: start_truth is False")
    lmax = int(d["lmax"])
    if not (np.array_equal(d["phi_initial"], d["phi_true_packed"])
            and np.array_equal(d["x0_start"][lmax - 2:], d["alm_true_packed"])):
        bad.append(f"{f}: start is not the truth")
    if "cl_phiphi_samples" not in d.files:
        bad.append(f"{f}: Block 4 was off")
print("\n".join(bad) if bad else "all 8 truth-start chains complete")
sys.exit(1 if bad else 0)
EOF

echo "================ 2. band-edge diagnostic ================"
.venv/bin/python -u scripts/diagnostics/diagnose_alm_band_edge.py --indir $TRUTHSTART --indir $CONTROL --split 0
rc=$?

echo "================ 3. paired vs the dense reference (r036 excluded) ================"
.venv/bin/python -u scripts/diagnostics/compare_band_edge_paired.py --ref $DENSE \
    --indir $CONTROL --indir $TRUTHSTART --exclude 36 --out $TRUTHSTART/compare_band_edge_paired.npz || rc=1
echo "================ 3b. the same with r036 ================"
.venv/bin/python -u scripts/diagnostics/compare_band_edge_paired.py --ref $DENSE \
    --indir $CONTROL --indir $TRUTHSTART || rc=1
echo "harvest exit status $rc"
exit $rc
