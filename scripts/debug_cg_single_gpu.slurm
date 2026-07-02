#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH -J debug_cg_1gpu
#SBATCH -o logs/debug_cg_1gpu_%j.out
#SBATCH -e logs/debug_cg_1gpu_%j.err

# Reproduces the exact production device layout (CUDA_VISIBLE_DEVICES=0 ->
# 1 GPU + N-1 CPU parts, as used by scripts/submit_cg_L300.slurm / job 11552530)
# to check whether the cross-device matvec bug (fixed in 66f169c for GPU/GPU
# splits) also affects GPU/CPU splits, which the existing regression test
# (tests/test_cg_matvec.py) and validation job 11552527 never exercised.

echo "===="
echo "Starting debug_cg_1gpu $SLURM_JOB_ID at $(date)"
echo "Running on nodes: $SLURM_NODELIST"
echo "===="

cd /cosma/apps/durham/dc-hick2/diffcmb
mkdir -p logs

export PYTHONPATH=$(pwd)/diffcmb
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
# Force a fresh TMPDIR: two earlier attempts on this same node (11555241,
# 11555248) crashed with a bizarre autograph error ("tf___ensure_tf_tensors()
# takes 1 positional argument but 2 were given") from inside a traced call to
# an unrelated method, while every smaller-scale repro (lmax=10/30, real
# data, GPU, up to 20 sph_parts) succeeded cleanly. Autograph writes its
# transpiled sources to $TMPDIR/__autograph_generated_file*.py, which is
# node-local and NOT job-scoped by default — if a stale/colliding cache
# entry from an earlier job on this node is being reused, a private TMPDIR
# rules that out.
export TMPDIR="/tmp/diffcmb_debugcg_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

.venv/bin/python scripts/debug_cg.py results/lmax300_nside256_gibbs_real_double

echo "===="
echo "Finished at $(date)"
echo "===="
