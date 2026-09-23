"""ROADMAP T0.1c: is the a_lm `[30,60)` power-rank offset a property of the
STATISTIC rather than of the sampler?

The coverage ensembles draw alm_true ~ N(0, C_l^fid) at a FIXED spectrum, but
Block 1 puts the flat improper prior on C_l. The truth is therefore not a draw
from the sampler's own prior, and the rank of the truth's binned a_lm power
among the posterior draws is not guaranteed uniform even for an exact sampler.
Under the flat prior the C_l posterior sits above the realized power, so
wherever the data do not pin the a_lm down, posterior a_lm power exceeds the
truth and the truth ranks LOW (mean_u < 0.5) -- the sign both exact-operator
ensembles show (0.27-0.30 in `[30,60)`, flat across chain quarters).

This script draws what a CORRECT sampler produces for that statistic, in the
full-sky diagonal Gaussian model the ensembles reduce to at fixed phi:

    a_j ~ N(0, C_l v_j),  d_j = a_j + n_j,  n_j ~ N(0, N_l v_j)

with v_j = 1 for m = 0 and 1/2 for each real/imag part at m > 0 (the packed
weighting of `model.compute_sl_np`). Its joint posterior under the flat
C_l prior is sampled EXACTLY and independently (no Markov chain):

    X = C_l + N_l ~ InvGamma(k_l/2 - 1, D_l/2) truncated to X > N_l,
    a_j | C_l, d ~ N(W d_j, W N_l v_j),   W = C_l / (C_l + N_l),

D_l = sum_j d_j^2 / v_j, k_l = packed dof, shape from
`invgamma_shape_for_spectrum` (never hardcoded).

Two noise levels:
  nominal    N_l = sigma^2 * 4 pi / N_pix (white pixel noise, no lensing)
  effective  N_l chosen so that W_l matches the per-l posterior variance
             fraction measured on the ensemble's own chains,
             W_l = <var_sweeps(a_j) / (C_l^true v_j)>. This carries the
             information the lensing (phi uncertainty) removes. It uses the
             chains' marginal variance, which includes C_l scatter, so it
             slightly overstates N_eff -- i.e. errs toward a LARGER null
             offset, so a surviving observed offset is conservative.

For each bin it prints the null mean_u with its standard error for the
ensemble size, the observed mean_u, and z = (obs - null) / se. |z| < 3 in
`[30,60)` under the effective null => the offset is the statistic, and the
three-block core is not rejected by it. |z| >> 3 => a sampler defect.

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/null_alm_power_rank_flat_prior.py \
      --indir results/analysis/ens_exact_l64_A3000_n30_nocl4 \
      --indir results/analysis/ens_exact_l64_A3000_n30 --n_null 2000
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper"))

from aggregate_coverage_ranks import binned_power, rank_of  # noqa: E402
from diagnose_calibration_stationarity import (  # noqa: E402
    chain_files,
    load_traces,
    rank_table,
)
from fig1_validation import ELL_BINS  # noqa: E402

from diffcmb.alm_utils import (  # noqa: E402
    invgamma_shape_for_spectrum,
    packed_dof_per_multipole,
    packed_sizes,
)
from diffcmb.samplers import _alm_index_lm  # noqa: E402

try:
    from scipy.stats import invgamma
except ImportError:  # pragma: no cover - guarded like the package
    invgamma = None

OBS_THIN = 10


def packed_layout(lmax):
    """(L_arr, v) for the packed alm vector: multipole and variance weight."""
    n_real, n_imag = packed_sizes(lmax)
    L_arr, m_arr = _alm_index_lm(lmax, n_real, n_imag)
    v = np.where(m_arr == 0, 1.0, 0.5)
    return L_arr, v


def nominal_noise(lmax, noisesig, nside):
    """White pixel noise sigma -> N_l = sigma^2 Omega_pix, flat in l."""
    n = np.zeros(lmax)
    n[2:] = noisesig ** 2 * 4.0 * np.pi / (12 * nside ** 2)
    return n


def effective_noise(files, lmax):
    """N_eff,l from the posterior variance fraction W_l measured on chains.

    W_l = mean over chains and components at l of var_sweeps(a_j)/(C_l v_j);
    W = C/(C+N) inverts to N = C (1 - W) / W. W is clipped to (0, 0.999].
    """
    L_arr, v = packed_layout(lmax)
    w_sum = np.zeros(lmax)
    for f in files:
        d = np.load(f, allow_pickle=True)
        cl = np.asarray(d["cl_true"], dtype=np.float64)[:lmax]
        var = np.var(d["alm_samples"][:, lmax - 2:], axis=0)
        frac = var / (cl[L_arr] * v)
        w_sum += np.bincount(L_arr, weights=frac, minlength=lmax) / np.maximum(
            np.bincount(L_arr, minlength=lmax), 1)
    w = np.clip(w_sum / len(files), 1e-6, 0.999)
    cl = np.asarray(np.load(files[0], allow_pickle=True)["cl_true"])[:lmax]
    n = np.zeros(lmax)
    n[2:] = cl[2:] * (1.0 - w[2:]) / w[2:]
    return n, w


def exact_posterior_draws(d, cl_noise, lmax, n_draws, rng):
    """Independent draws of the packed a_lm from the flat-C_l joint posterior."""
    if invgamma is None:
        raise ImportError("scipy is required")
    L_arr, v = packed_layout(lmax)
    shape = invgamma_shape_for_spectrum(lmax, a0=-1.0)
    D = np.bincount(L_arr, weights=d ** 2 / v, minlength=lmax)
    X = np.zeros((n_draws, lmax))
    for L in range(2, lmax):
        tail = invgamma.sf(cl_noise[L], shape[L], scale=D[L] / 2.0)
        u = rng.uniform(0.0, tail, size=n_draws)
        X[:, L] = invgamma.isf(u, shape[L], scale=D[L] / 2.0)
    C = np.maximum(X - cl_noise[None, :], 0.0)
    W = np.divide(C, X, out=np.zeros_like(C), where=X > 0)[:, L_arr]
    return W * d[None, :] + np.sqrt(W * cl_noise[L_arr] * v) * rng.standard_normal(
        (n_draws, d.size))


def null_ranks(cl, cl_noise, lmax, n_null, n_draws, seed):
    """{(lo,hi): ranks} of the truth's binned power, one per null realization."""
    rng = np.random.default_rng(seed)
    L_arr, v = packed_layout(lmax)
    sd_sig = np.sqrt(cl[L_arr] * v)
    sd_noise = np.sqrt(cl_noise[L_arr] * v)
    bins = [(lo, min(hi, lmax)) for lo, hi in ELL_BINS if lo < min(hi, lmax)]
    out = {b: [] for b in bins}
    for _ in range(n_null):
        a = sd_sig * rng.standard_normal(L_arr.size)
        d = a + sd_noise * rng.standard_normal(L_arr.size)
        post = exact_posterior_draws(d, cl_noise, lmax, n_draws, rng)
        for lo, hi in bins:
            out[(lo, hi)].append(rank_of(binned_power(a, L_arr, lo, hi),
                                         binned_power(post, L_arr, lo, hi)))
    return {b: np.asarray(r) for b, r in out.items()}


def observed_mean_u(files, thin=OBS_THIN):
    """{(lo,hi): (mean_u, n_chains)} for alm_power, as the T0.1 diagnostic."""
    table = rank_table(load_traces(files), thin, "full")
    out = {}
    for (tag, lo, hi), (ranks, nd) in table.items():
        if tag == "alm_power":
            out[(lo, hi)] = (float(np.mean((ranks + 0.5) / (nd + 1.0))), len(ranks))
    return out, nd


def report(indir, n_null, seed):
    files = chain_files(indir)
    d0 = np.load(files[0], allow_pickle=True)
    lmax, nside = int(d0["lmax"]), int(d0["nside"])
    cl = np.asarray(d0["cl_true"], dtype=np.float64)[:lmax]
    obs, nd = observed_mean_u(files)
    n_eff, w = effective_noise(files, lmax)
    print(f"\n######## {indir}: {len(files)} chains, lmax {lmax}, "
          f"{nd + 1} draws/chain at thin {OBS_THIN} ########")
    print("  posterior variance fraction W_l (chains) at l = 5, 20, 45, 62: "
          + ", ".join(f"{w[L]:.3f}" for L in (5, 20, 45, 62) if L < lmax))
    results = {}
    for mode, cl_noise in (("nominal", nominal_noise(lmax, float(d0["noisesig"]), nside)),
                           ("effective", n_eff)):
        # same draw count and u = (r+0.5)/(nd+1) as the observed ranks
        ranks = null_ranks(cl, cl_noise, lmax, n_null, nd + 1, seed)
        print(f"\n  {mode} null ({n_null} exact realizations):")
        print("    bin        null mean_u  se(N_obs)   observed   z")
        for b, r in ranks.items():
            u = (r + 0.5) / (nd + 1.0)
            o, n_obs = obs[b]
            se = u.std(ddof=1) / np.sqrt(n_obs)
            z = (o - u.mean()) / se
            print(f"    [{b[0]:2d},{b[1]:2d})    {u.mean():.3f}       {se:.3f}      "
                  f"{o:.3f}    {z:+.2f}")
            results[f"{mode}_{b[0]}_{b[1]}_u"] = u
            results[f"{mode}_{b[0]}_{b[1]}_obs"] = o
        results[f"{mode}_cl_noise"] = cl_noise
    out = os.path.join(indir, "null_alm_power_rank_flat_prior.npz")
    np.savez(out, w=w, **results)
    print(f"\nSaved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indir", action="append", required=True)
    ap.add_argument("--n_null", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260924)
    args = ap.parse_args()
    for indir in args.indir:
        report(indir, args.n_null, args.seed)


if __name__ == "__main__":
    main()
