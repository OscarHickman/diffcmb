#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska,cosma8-ska2,cosma7-shm,bluefield1
#SBATCH --exclude=mad01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH -J realdata_summary
#SBATCH -o logs/realdata_summary_%j.out
#SBATCH -e logs/realdata_summary_%j.err
# ROADMAP T1.2a: realdata_summary.py on each real-data run directory in DIRS
# (space-separated), writing <dir>/summary.npz. Submit with
# --dependency=afterany:<chain arrays> --kill-on-invalid-dep=yes.
set -u
cd /cosma/apps/durham/dc-hick2/diffcmb
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/diffcmb
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 NUMBA_NUM_THREADS=2
rc=0
for d in $DIRS; do
  echo "================ $d"
  .venv/bin/python -u scripts/realdata/realdata_summary.py --chains $d --out $d/summary || rc=1
done
exit $rc
