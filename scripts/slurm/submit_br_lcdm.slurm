#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska,cosma8-ska2,cosma7-shm,bluefield1
#SBATCH --exclude=mad01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH -J br_lcdm
#SBATCH -o logs/br_lcdm_%j.out
#SBATCH -e logs/br_lcdm_%j.err
# ROADMAP T1.4: the Blackwell-Rao (A, n_s) likelihood from the joint chains (scripts/br_lcdm.py).
# 1. calibration on the flat-C_l-prior full-sky ensemble (every sky's spectrum is the fiducial,
#    so (A, n_s) = (1, 0.965) is every sky's truth; the per-l product is exact there),
#    --target lcdm: the posterior CDF at the truth must be uniform;
# 2. calibration under the mask: the strict real-sky ensemble (Planck mask, beam, ecliptic
#    noise; truths drawn from the nu = 30 prior around the fiducial), --target hier;
# 3. real data: SMICA and PR4 SEVEM at A_phi 1, each noise-marginalised over its 8 draws,
#    --target lcdm (the physical comparison with Planck's fit).
# Reads results/analysis (/cosma5), so /cosma5-mounting partitions only.
set -u
cd /cosma/apps/durham/dc-hick2/diffcmb
mkdir -p logs results/analysis/br_lcdm
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/diffcmb
NT=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT DUCC0_NUM_THREADS=$NT
export RAYON_NUM_THREADS=$NT NUMBA_NUM_THREADS=$NT TF_NUM_INTRAOP_THREADS=$NT TF_NUM_INTEROP_THREADS=1
A=results/analysis
.venv/bin/python -u scripts/br_lcdm.py --ensemble $A/ens_exact_l64_A3000_n30_nu30_long --target lcdm \
    --out $A/br_lcdm/d5_calibration || exit 1
.venv/bin/python -u scripts/br_lcdm.py --ensemble $A/ens_realsky_l64_A3000_n30_nu30_strict --target hier \
    --out $A/br_lcdm/realsky_strict_calibration || exit 1
for MAP in smica sevemR4; do
  R=$A/realdata_${MAP}_l64_A1_n30_clnu30
  .venv/bin/python -u scripts/br_lcdm.py --chains $R ${R}_nr1 ${R}_nr2 ${R}_nr3 ${R}_nr4 ${R}_nr5 ${R}_nr6 ${R}_nr7 \
      --target lcdm --out $A/br_lcdm/${MAP}_A1_pooled8 || exit 1
done
echo done
