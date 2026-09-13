#!/bin/bash
#SBATCH -A durham
#SBATCH -p cosma5
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=128G
#SBATCH --time=06:00:00
#SBATCH -J blind192
#SBATCH -o logs/blind192_%j.out
#SBATCH -e logs/blind192_%j.err

# 2026-09-13. Lensing-blind C_l^TT baseline at lmax=192, for the higher-lmax
# bias-reduction figure (ROADMAP Todo section 2 / decision S4: spend scale on
# the bias reduction, not on the null correlation).
#
# WHY lmax=192 AND NOT 256. The effect grows with l (-5.4% already at
# [100,128)), so more l is more figure. But the AWARE side is the cost: at
# lmax=128 it runs ~49 s/sweep for 2300 sweeps (~31h), and cost scales roughly
# as lmax^3 for the matrix-free SHT path, so 256 would be ~8x (250h+, far past
# the 72h cap and needing chained resumes) while 192 is ~3.4x (~37h for a
# shortened chain) and fits in one job. A companion timing pilot measures this
# rather than trusting the scaling estimate.
#
# The blind side has NO phi block and runs at ~0.1 s/sweep at lmax=128, so it is
# cheap even at 192 -- run it first and independently, exactly as at lmax=128
# (job 11980570 took 4 minutes).
#
# Same seed (0) as the aware side will use, so the two analyse the SAME sky;
# compare_cl_bias_reduction.py asserts their alm_true_packed agree before
# comparing and fails loudly otherwise.

cd /cosma/apps/durham/dc-hick2/diffcmb
mkdir -p logs results/analysis
export TMPDIR="/tmp/dc-hick2-${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb
export OMP_NUM_THREADS=16 TF_ENABLE_ONEDNN_OPTS=0

echo "=== lensing-blind baseline lmax=192, job ${SLURM_JOB_ID} ==="
free -h | head -2

.venv/bin/python -u scripts/run_lensing_blind_baseline.py \
    --lmax 192 --nside 192 --noisesig 1.0 \
    --n_burnin 500 --n_samples 3000 --seed 0 \
    --checkpoint_path results/analysis/lensing_blind_baseline_lmax192_ckpt.npz \
    --checkpoint_every 100 \
    --out results/analysis/lensing_blind_baseline_lmax192.npz
echo "Exited with status $?"
