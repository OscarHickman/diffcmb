#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=192G
#SBATCH --time=08:00:00
#SBATCH -J aware192pilot
#SBATCH -o logs/aware192pilot_%j.out
#SBATCH -e logs/aware192pilot_%j.err

# 2026-09-13. TIMING / FEASIBILITY PILOT for the lensing-aware chain at
# lmax=192 -- measure, do not assume, before committing a multi-day job.
#
# WHAT THIS DECIDES. The full-length run's cost is estimated by scaling
# lmax=128's measured 49 s/sweep by ~lmax^3 (matrix-free SHT), giving ~165
# s/sweep. That estimate has never been checked at this scale, and the whole
# point of a pilot here is that getting it wrong means discovering it 72h into a
# walltime. 250 sweeps is enough to read a stable s/sweep and peak RSS, and to
# confirm the MAP start behaves (alm cosine well above 0, phi ratio O(1)) --
# the cold-start failure mode that broke two earlier campaigns.
#
# EXPECT A NO-GO ON THE EQUILIBRATION GATE and ignore it: phi does not
# equilibrate at low l even at lmax=128 (jobs 11966631, 11984844, worst lag-1
# 0.964-0.975), and the bias-reduction figure is built from mid/high l where it
# decorrelates within 10-50 lags. The gate is not the acceptance criterion for
# THIS run; s/sweep, memory and the MAP-start diagnostics are.
#
# Config otherwise matches the lmax=128 aware chains exactly (HMC -- NOT the
# script's mclmc default, which is a closed NO-GO; phi_mass_matrix='prior';
# phi_n_lfs=240; map_steps=2000/map_lr=0.01; seed=0 to match the blind side).

cd /cosma/apps/durham/dc-hick2/diffcmb
mkdir -p logs results/analysis
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=16 TF_ENABLE_ONEDNN_OPTS=0

echo "=== lmax=192 aware TIMING PILOT, job ${SLURM_JOB_ID} ==="
echo "node $(hostname)"; free -h | head -2

.venv/bin/python -u scripts/pilot_coverage_equilibration.py \
    --lmax 192 --nside 192 --noisesig 1.0 \
    --n_burnin 50 --map_steps 2000 --map_lr 0.01 \
    --n_samples 200 --n_lfs 10 \
    --phi_sampler hmc --phi_mass_matrix prior --phi_n_lfs 240 \
    --seed 0 \
    --checkpoint_path results/analysis/aware192_pilot_ckpt.npz \
    --checkpoint_every 25 \
    --out results/analysis/aware192_pilot.npz
echo "Exited with status $?"
