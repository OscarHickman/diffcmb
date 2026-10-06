"""ROADMAP T0.1 (2026-09-27): where inside the top a_lm bin does the power-rank
offset live, and does it survive thinning and a direct calibration test?

At N = 48 the a_lm power rank in `[lmax-4, lmax)` sits at z +3.45 against the
effective exact-sampler null (`null_alm_power_rank_flat_prior.py`), and it
reproduces on fresh skies. This script breaks that statistic down, reusing the
same null model (flat C_l prior, diagonal Gaussian, nominal or chain-calibrated
effective noise) so every observed number has an exact-sampler reference:

  a. per multipole l = lmax-4..lmax-1, per m group (m = 0, 0 < m <= l/2,
     m > l/2) and real vs imaginary parts, with a reference bin
     [lmax-8, lmax-4) just below;
  b. the rank at several thinnings, and by chain quarter (stationarity);
  c. the effective-noise edge: f_l per l, and a null-free calibration check
     over every saved sweep -- per-mode z = (a_true - posterior mean) / posterior sd and the shrinkage
     slope <mean * a_true> / <a_true^2> per l, both against the same exact
     model. sd(z) > 1 means posteriors too narrow; a slope below the model's
     W_l means posterior means shrunk harder than an exact sampler's.

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/diagnose_alm_band_edge.py \
      --indir results/analysis/ens_exact_l64_A3000_n30_nu30_long --thin 50 --thin 100
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper"))

from diagnose_calibration_stationarity import chain_files  # noqa: E402
from null_alm_power_rank_flat_prior import (  # noqa: E402
    effective_noise,
    exact_posterior_draws,
    nominal_noise,
)

from diffcmb.alm_utils import packed_sizes  # noqa: E402
from diffcmb.samplers import _alm_index_lm  # noqa: E402

N_QUARTERS = 4
Z_NULL_DRAWS = 200


def edge_groups(lmax):
    """{name: boolean mask over the packed alm} for the band-edge breakdown."""
    n_real, n_imag = packed_sizes(lmax)
    L_arr, m_arr = _alm_index_lm(lmax, n_real, n_imag)
    is_real = np.arange(L_arr.size) < n_real
    top = L_arr >= lmax - 4
    groups = {f"bin [{lmax - 8},{lmax - 4})": (L_arr >= lmax - 8) & ~top,
              f"bin [{lmax - 4},{lmax})": top}
    for L in range(lmax - 4, lmax):
        groups[f"l = {L}"] = L_arr == L
    groups["top: m = 0"] = top & (m_arr == 0)
    groups["top: 0 < m <= l/2"] = top & (m_arr > 0) & (2 * m_arr <= L_arr)
    groups["top: m > l/2"] = top & (2 * m_arr > L_arr)
    groups["top: real parts"] = top & is_real
    groups["top: imag parts"] = top & ~is_real
    return groups


def u_of(truth_power, draw_power):
    """Rank u = (#draws below truth + 0.5) / (n + 1), the fig1 grid."""
    return (np.sum(draw_power < truth_power) + 0.5) / (draw_power.size + 1.0)


def load(files, lmax):
    """[(alm draws (n_sweep, n_alm), alm_true)] for every chain."""
    out = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        out.append((np.asarray(d["alm_samples"][:, lmax - 2:], dtype=np.float64),
                    np.asarray(d["alm_true_packed"], dtype=np.float64)))
    return out


def observed_u(chains, mask, thin, segment=None):
    """Per-chain u of the truth's group power; segment = (k, n_seg) or None."""
    us = []
    for alm, truth in chains:
        s = alm if segment is None else np.array_split(alm, segment[1])[segment[0]]
        s = s[::thin]
        us.append(u_of(np.mean(truth[mask] ** 2), np.mean(s[:, mask] ** 2, axis=1)))
    return np.asarray(us)


def null_u(cl, cl_noise, lmax, groups, n_null, n_draws, seed):
    """{group: u array over n_null exact realizations}, one shared set of draws."""
    rng = np.random.default_rng(seed)
    n_real, n_imag = packed_sizes(lmax)
    L_arr, m_arr = _alm_index_lm(lmax, n_real, n_imag)
    v = np.where(m_arr == 0, 1.0, 0.5)
    out = {g: np.empty(n_null) for g in groups}
    for i in range(n_null):
        a = np.sqrt(cl[L_arr] * v) * rng.standard_normal(L_arr.size)
        d = a + np.sqrt(cl_noise[L_arr] * v) * rng.standard_normal(L_arr.size)
        post = exact_posterior_draws(d, cl_noise, lmax, n_draws, rng)
        for g, m in groups.items():
            out[g][i] = u_of(np.mean(a[m] ** 2), np.mean(post[:, m] ** 2, axis=1))
    return out


def z_and_slope(pairs, L_arr, lmax):
    """Per-l sd of z = (truth - mean)/sd and shrinkage slope <mean a>/<a^2>."""
    zsq = np.zeros(lmax)
    cross = np.zeros(lmax)
    tsq = np.zeros(lmax)
    for draws, truth in pairs:
        mu = draws.mean(axis=0)
        sd = draws.std(axis=0, ddof=1)
        zsq += np.bincount(L_arr, weights=((truth - mu) / sd) ** 2, minlength=lmax)
        cross += np.bincount(L_arr, weights=mu * truth, minlength=lmax)
        tsq += np.bincount(L_arr, weights=truth ** 2, minlength=lmax)
    count = np.bincount(L_arr, minlength=lmax) * len(pairs)
    return np.sqrt(zsq / np.maximum(count, 1)), cross / np.maximum(tsq, 1e-300)


def cl_ratio_per_chain(f, lmax, n_seg=N_QUARTERS):
    """(n_seg, lmax) <C_l> / (S_true,l / (k_l - 4)) per chain segment, one chain."""
    from diffcmb.alm_utils import packed_dof_per_multipole

    n_real, n_imag = packed_sizes(lmax)
    L_arr, m_arr = _alm_index_lm(lmax, n_real, n_imag)
    v = np.where(m_arr == 0, 1.0, 0.5)
    k = packed_dof_per_multipole(lmax)
    d = np.load(f, allow_pickle=True)
    C = np.exp(np.asarray(d["alm_samples"][:, :lmax - 2], dtype=np.float64))
    t = np.asarray(d["alm_true_packed"], dtype=np.float64)
    ref = np.bincount(L_arr, weights=t ** 2 / v, minlength=lmax)[2:] / (k[2:] - 4)
    out = np.zeros((n_seg, lmax))
    for q, seg in enumerate(np.array_split(C, n_seg)):
        out[q, 2:] = seg.mean(axis=0) / ref
    return out


def cl_ratio_by_quarter(files, lmax, n_seg=N_QUARTERS):
    """(n_seg, lmax) mean over chains of <C_l> / (S_true,l / (k_l - 4)) per segment.

    An exact sampler sits at a CONSTANT level across segments: ~1 where the
    data pin the a_lm, above 1 where noise inflates E[S|d] (the figure 2
    effect). A trend across segments is non-stationarity (2026-09-27: C_63
    climbed 0.934 -> 0.947 from the low MAP start, against 0.99 in the dense
    exact reference on the same skies).
    """
    return np.mean([cl_ratio_per_chain(f, lmax, n_seg) for f in files], axis=0)


def null_pairs(cl, cl_noise, lmax, n_real_sky, seed):
    """Exact-sampler (draws, truth) pairs for the z / slope reference."""
    rng = np.random.default_rng(seed)
    n_real, n_imag = packed_sizes(lmax)
    L_arr, m_arr = _alm_index_lm(lmax, n_real, n_imag)
    v = np.where(m_arr == 0, 1.0, 0.5)
    pairs = []
    for _ in range(n_real_sky):
        a = np.sqrt(cl[L_arr] * v) * rng.standard_normal(L_arr.size)
        d = a + np.sqrt(cl_noise[L_arr] * v) * rng.standard_normal(L_arr.size)
        pairs.append((exact_posterior_draws(d, cl_noise, lmax, Z_NULL_DRAWS, rng), a))
    return pairs


def zline(obs, null):
    se = null.std(ddof=1) / np.sqrt(obs.size)
    return f"{obs.mean():.3f}  {null.mean():.3f}  {(obs.mean() - null.mean()) / se:+6.2f}"


def report(indir, thins, n_null, seed, subsets):
    files = chain_files(indir)
    d0 = np.load(files[0], allow_pickle=True)
    lmax, nside = int(d0["lmax"]), int(d0["nside"])
    cl = np.asarray(d0["cl_true"], dtype=np.float64)[:lmax]
    chains = load(files, lmax)
    n_sweep = chains[0][0].shape[0]
    groups = edge_groups(lmax)
    n_eff, w = effective_noise(files, lmax)
    noises = {"nominal": nominal_noise(lmax, float(d0["noisesig"]), nside),
              "effective": n_eff}
    print(f"\n######## {indir}: {len(files)} chains, lmax {lmax}, {n_sweep} sweeps ########")
    ells = list(range(lmax - 8, lmax))
    print("  f_l = 1 - W_l (chains) at l = " + ", ".join(str(L) for L in ells))
    print("      " + ", ".join(f"{1 - w[L]:.3f}" for L in ells))
    print("  N_eff / N_nominal:  " + ", ".join(
        f"{n_eff[L] / noises['nominal'][L]:.2f}" for L in ells))
    saved = {"w": w, "n_eff": n_eff}
    for thin in thins:
        nd = len(range(0, n_sweep, thin))
        nulls = {k: null_u(cl, nz, lmax, groups, n_null, nd, seed) for k, nz in noises.items()}
        print(f"\n  thin {thin} ({nd} draws/chain); columns: observed, null mean, z")
        print(f"    {'group':22s} {'nominal':>22s}   {'effective':>22s}"
              + "".join(f"   {name:>18s}" for name in subsets))
        for g, m in groups.items():
            obs = observed_u(chains, m, thin)
            line = f"    {g:22s} {zline(obs, nulls['nominal'][g]):>22s}   " \
                   f"{zline(obs, nulls['effective'][g]):>22s}"
            for _name, idx in subsets.items():
                sub = obs[idx[idx < obs.size]]
                if sub.size:
                    e = nulls["effective"][g]
                    line += f"   {(sub.mean() - e.mean()) / (e.std(ddof=1) / np.sqrt(sub.size)):+18.2f}"
            print(line)
            saved[f"obs_u_thin{thin}_{g}"] = obs
            for k in noises:
                saved[f"null_{k}_u_thin{thin}_{g}"] = nulls[k][g]
    top = groups[f"bin [{lmax - 4},{lmax})"]
    thin = thins[0]
    print(f"\n  stationarity of bin [{lmax - 4},{lmax}) at thin {thin}: mean u by chain quarter")
    for q in range(N_QUARTERS):
        obs = observed_u(chains, top, thin, segment=(q, N_QUARTERS))
        print(f"    quarter {q + 1}: {obs.mean():.3f} +- {obs.std(ddof=1) / np.sqrt(obs.size):.3f}")
    ratio = cl_ratio_by_quarter(files, lmax)
    print("\n  posterior C_l / (S_true/(k-4)) by chain quarter (exact sampler: flat in quarter)")
    for L in ells:
        print(f"    l = {L}:  " + "  ".join(f"{ratio[q, L]:.3f}" for q in range(N_QUARTERS)))
    saved["cl_ratio_by_quarter"] = ratio
    n_real, n_imag = packed_sizes(lmax)
    L_arr, _ = _alm_index_lm(lmax, n_real, n_imag)
    # every sweep: a thinned sd from ~24 draws would itself inflate sd(z)
    sd_obs, slope_obs = z_and_slope(chains, L_arr, lmax)
    print("\n  calibration per l (all sweeps): sd(z) observed | exact null (nominal, effective);"
          " shrinkage slope observed | W_l model (nominal, effective)")
    ref = {k: z_and_slope(null_pairs(cl, nz, lmax, len(chains), seed + 1), L_arr, lmax)
           for k, nz in noises.items()}
    for L in ells:
        print(f"    l = {L}:  sd(z) {sd_obs[L]:.3f} | {ref['nominal'][0][L]:.3f}, "
              f"{ref['effective'][0][L]:.3f}    slope {slope_obs[L]:.3f} | "
              f"{ref['nominal'][1][L]:.3f}, {ref['effective'][1][L]:.3f}")
    saved.update(sd_z_obs=sd_obs, slope_obs=slope_obs,
                 sd_z_nominal=ref["nominal"][0], slope_nominal=ref["nominal"][1],
                 sd_z_effective=ref["effective"][0], slope_effective=ref["effective"][1])
    out = os.path.join(indir, "diagnose_alm_band_edge.npz")
    np.savez(out, **saved)
    print(f"\nSaved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indir", action="append", required=True)
    ap.add_argument("--thin", type=int, action="append",
                    help="thinnings to test; the first is used for quarters and z (default 50, 100, 150)")
    ap.add_argument("--n_null", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--split", type=int, default=24,
                    help="also report the effective z on chains [0, split) and [split, N)")
    args = ap.parse_args()
    thins = args.thin or [50, 100, 150]
    for indir in args.indir:
        n = len(chain_files(indir))
        subsets = {}
        if 0 < args.split < n:
            subsets = {f"z eff r<{args.split}": np.arange(args.split),
                       f"z eff r>={args.split}": np.arange(args.split, n)}
        report(indir, thins, args.n_null, args.seed, subsets)


if __name__ == "__main__":
    main()
