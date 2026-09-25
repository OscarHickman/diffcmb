#!/bin/bash -l
#SBATCH -J rescore_phi_equil
#SBATCH -o logs/rescore_phi_equil_%j.out
#SBATCH -e logs/rescore_phi_equil_%j.err
#SBATCH -A durham
#SBATCH -p cosma5,cosma8-shm,cosma8-shm2,cosma8-shm3,cosma8-ska,bluefield1
#SBATCH -t 00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
# Not dine2: results/analysis is a symlink into /cosma5, which dine2 does not mount
# (ROADMAP 2026-09-24). Memory sized from past MaxRSS instead of a whole node.

# Pure re-analysis of already-saved pilot chains -- no sampling, no TF.
# Separates "slow but stationary" from "not equilibrated" (see the script
# docstring); decides whether the coverage ensemble is feasible on chains
# the project already has.

export TMPDIR="/tmp/rescore_phi_equil_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
export PYTHONPATH=diffcmb

cd /cosma/apps/durham/dc-hick2/diffcmb

.venv/bin/python scripts/rescore_phi_equilibration.py \
    results/analysis/pilot_coverage_lmax64_hmc_ordfix_n3300.npz \
    results/analysis/pilot_coverage_lmax64_block_cl4_n3300.npz \
    results/analysis/pilot_coverage_lmax128_block_n3300.npz

rm -rf "$TMPDIR"
