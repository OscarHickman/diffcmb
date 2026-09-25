#!/bin/bash
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -J diagnose_phi_hessian_l10_30
#SBATCH -o logs/diagnose_phi_hessian_l10_30_%j.out
#SBATCH -e logs/diagnose_phi_hessian_l10_30_%j.err

# ROADMAP.md NEXT SESSION (2026-09-02): scoped follow-up to job 11912088's
# harvest. The doubled-trajectory ensemble moved the strict C_L^phiphi SBC
# rank from 0.3802 to 0.4196 but the failure is now concentrated almost
# entirely in ell-bin [10,30). This tests whether cross-L Hessian coupling
# (the mechanism diagnose_phi_hessian_coupling.py found significant in
# 2026-08-17, pre-ordering-fix) still implicates [10,30) under the CURRENT
# production configuration (post ordering fix, post dof fix, proper prior,
# job 11912088's own chain states) -- see
# scripts/diagnose_phi_hessian_coupling_l10_30.py's docstring for the full
# reasoning and how this differs from the original script.
#
# No new MCMC: reuses job 11912088's saved chain_r*.npz files and probes
# analytic-gradient finite differences at a few of their states. A 1-probe,
# 1-chain-point, 1-realization smoke test ran in well under a minute on the
# login node (2026-09-02); this full run (3 realizations x 4 chain points x
# 3 comparison bins x 6 probes) is a few hundred such evals, so this
# walltime is generous.

# results/analysis is a symlink into /cosma5, which dine2 does not mount (ROADMAP
# 2026-09-24). Fail loudly here rather than at the first write.
if [ ! -d /cosma/apps/durham/dc-hick2/diffcmb/results/analysis/ ]; then
    echo "results/analysis (/cosma5) is not mounted on $(hostname); relocate outputs before running on dine2" >&2
    exit 1
fi
REPO_DIR="/cosma/apps/durham/dc-hick2/diffcmb"
cd $REPO_DIR

mkdir -p logs results/analysis

PYTHON=".venv/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)/diffcmb:$(pwd)/scripts
export OMP_NUM_THREADS=8
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "phi-block Hessian cross-L coupling diagnosis, targeted at [10,30) (job 11912088's chains)"
echo "Using Python: $($PYTHON --version 2>&1)"

$PYTHON -u scripts/diagnose_phi_hessian_coupling_l10_30.py \
    --realizations 0,1,2,3,4,5 \
    --n_chain_points 4 \
    --n_probes 6 \
    --out results/analysis/diagnose_phi_hessian_coupling_l10_30.npz
