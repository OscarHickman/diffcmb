"""Figure 4 check (ROADMAP T0.2): are the cells above the permutation null real?

Prints each (C_l bin x C_L^phiphi bin) cell's pooled r, its 95% permutation
null, and a chain-level z (mean of per-chain r over its SEM across chains),
which is robust to within-chain autocorrelation that the sweep-permutation
null ignores. Run at the figure's thin and at a much larger one.

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/inspect_fig4_cells.py [--indir DIR]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "paper")]
import argparse

from plot_joint_cl_clpp_posterior import (
    CL_BINS,
    CLPP_BINS,
    load_pairs,
    permutation_null,
    pooled_corr,
    standardise,
)

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--indir", default="results/analysis/ens_exact_l64_A3000_n30_nu30_long")
ind = ap.parse_args().indir
rng = np.random.default_rng(0)
for thin in (45, 150):
    files, pairs = load_pairs(ind, thin)
    corr = pooled_corr(pairs)
    null = np.percentile(np.abs(permutation_null(pairs, 2000, rng)), 95, axis=0)
    # per-chain correlations -> chain-level t-test (robust to autocorrelation)
    per = np.array([[[np.mean(standardise(cl)[:, i] * standardise(pp)[:, j]) for j in range(4)]
                     for i in range(4)] for cl, pp, _ in pairs])
    m, se = per.mean(0), per.std(0, ddof=1) / np.sqrt(len(per))
    print(f"\n== thin {thin}: {pairs[0][0].shape[0]} draws/chain, {len(pairs)} chains")
    print("cell (Cl x Clpp)      r     null95  ratio   chain-mean  z_chain  frac>0")
    for i in range(4):
        for j in range(4):
            star = "*" if abs(corr[i, j]) > null[i, j] else " "
            print(f"{star} {CL_BINS[i]} x {CLPP_BINS[j]}  {corr[i,j]:+.3f}  {null[i,j]:.3f}  {corr[i,j]/null[i,j]:+.2f}   "
                  f"{m[i,j]:+.3f}  {m[i,j]/se[i,j]:+.2f}   {np.mean(per[:,i,j]>0):.2f}")
