#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=192G
#SBATCH --time=72:00:00
#SBATCH -J aware192
#SBATCH -o logs/aware192_%j.out
#SBATCH -e logs/aware192_%j.err

# 2026-09-13. FULL lensing-aware chain at
# lmax=192 for the higher-lmax C_l^TT bias-reduction figure.
#
# SIZED FROM THE PILOT (job 11986726): measured 100 s/sweep on cosma5 with 16
# cores -- ~2x lmax=128, cheaper than the lmax^3 estimate (3.4x). MAP start
# healthy there (alm cosine +0.982, phi power ratio 1.037). 400 burn-in + 1500
# samples = 1900 sweeps ~ 53h, inside the 72h cap; checkpoint_every=50 so a
# walltime overrun resumes rather than restarts. 1500 samples is ample: at
# lmax=128, 1900 samples gave blocking SEMs of ~0.0003 on a -5% effect.
#
# HARVEST: compare against lensing_blind_baseline_lmax192.npz (job 11986725,
# same seed=0) with compare_cl_bias_reduction.py --lmax 192. NOTE its BINS and
# UNRELIABLE set are hard-coded for lmax=128 and must be extended to cover
# [128,192) before this is run.
#
# (Original pilot header follows.)
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

echo "=== lmax=192 aware FULL RUN, job ${SLURM_JOB_ID} ==="
echo "node $(hostname)"; free -h | head -2

.venv/bin/python -u scripts/pilot_coverage_equilibration.py \
    --lmax 192 --nside 192 --noisesig 1.0 \
    --n_burnin 400 --map_steps 2000 --map_lr 0.01 \
    --n_samples 1500 --n_lfs 10 \
    --phi_sampler hmc --phi_mass_matrix prior --phi_n_lfs 240 \
    --seed 0 \
    --checkpoint_path results/analysis/aware192_ckpt.npz \
    --checkpoint_every 50 \
    --out results/analysis/pilot_coverage_lmax192_hmc.npz
echo "Exited with status $?"
