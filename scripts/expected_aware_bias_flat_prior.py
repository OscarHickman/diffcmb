"""Figure 2 check (ROADMAP T0.2): the aware C_l bias an EXACT sampler shows.

Under the flat C_l prior the chain's posterior-mean C_l is E[S_l(a)|d]/(k_l-4),
while fig2's reference is S_l(a_true)/(k_l-4). Noise makes E[S|d] > S_true, so
an exact sampler's aware residual is positive (the same effect as figure 1's
a_lm rank offset). For each real sky, draw d = a_true + n at nominal and at
chain-calibrated effective noise, take exact posterior draws
(null_alm_power_rank_flat_prior.exact_posterior_draws), and form fig2's bins.
The two noise models bracket the expectation.

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/expected_aware_bias_flat_prior.py [--indir DIR]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "paper")]
import argparse

import null_alm_power_rank_flat_prior as nul
from diagnose_calibration_stationarity import chain_files
from fig2_bias_reduction import BINS

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--indir", default="results/analysis/ens_exact_l64_A3000_n30_nu30_long")
ind = ap.parse_args().indir
files = chain_files(ind)
d0 = np.load(files[0])
lmax = int(d0["lmax"])
L, v = nul.packed_layout(lmax)
k = np.bincount(L, minlength=lmax).astype(float)
n_eff, _ = nul.effective_noise(files, lmax)
n_nom = nul.nominal_noise(lmax, float(d0["noisesig"]), int(d0["nside"]))
rng = np.random.default_rng(1)


def bias(a, noise, n_d=20, n_post=200):
    S_true = np.bincount(L, weights=a**2 / v, minlength=lmax)
    out = []
    for _ in range(n_d):
        d = a + np.sqrt(noise[L] * v) * rng.standard_normal(a.size)
        post = nul.exact_posterior_draws(d, noise, lmax, n_post, rng)
        ES = np.array([np.bincount(L, weights=p**2 / v, minlength=lmax) for p in post]).mean(0)
        row = []
        for lo, hi in BINS:
            ell = np.arange(lo, min(hi, lmax))
            w = 2 * ell + 1
            row.append(np.sum(ES[ell] / (k[ell] - 4) * w) / np.sum(S_true[ell] / (k[ell] - 4) * w) - 1)
        out.append(row)
    return np.mean(out, 0) * 100, np.std(out, 0) * 100


for name, noise in (("effective", n_eff), ("nominal", n_nom)):
    per = []
    sd = []
    for f in files:
        a = np.asarray(np.load(f)["alm_true_packed"], float)
        m, s = bias(a, noise)
        per.append(m)
        sd.append(s)
    per = np.array(per)
    sd = np.array(sd)
    print(f"\n{name} noise: expected aware bias (%) over 24 skies; per-sky data scatter")
    for i, (lo, hi) in enumerate(BINS):
        print(
            f"  [{lo:2d},{hi:2d})  expected mean {per[:, i].mean():+.3f}   per-sky noise sd {np.sqrt((sd[:, i]**2).mean()):.3f}"
            f"  -> SEM(24) {np.sqrt((sd[:, i]**2).mean())/np.sqrt(24):.3f}"
        )
