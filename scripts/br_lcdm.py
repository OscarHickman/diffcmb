"""Low-l LCDM parameters from the joint chains' a_lm samples (ROADMAP T1.4).

A Blackwell-Rao (BR) likelihood (Chu et al. 2005; the Commander low-l likelihood) built
from this work's chains. Every chain stores full-sky, unlensed a_lm samples, drawn
jointly with phi, so the likelihood has the lensing marginalised and the mask filled by
the sampler. Per multipole, an importance-sampling identity exact in expectation,

    L_l(C) = mean_i p(S_il | C) / p_chain(S_il),      ln L(theta) = sum_l ln L_l(C_l(theta)),

with a_i ~ p(a | d) under the chain's own C_l prior, whose marginal is divided out; both
factors depend on a only through S_l = sum_m |a_lm|^2 (k_l = 2l+1 real dof):

    ln p(S | C)       = -k/2 ln C - S / (2C)                         (target "lcdm")
    ln p(S | C^fid)   = nu/2 ln C^fid - (k+nu)/2 ln(S + nu C^fid)      (target "hier": C_l ~
                         InvGamma(nu/2, nu C^fid/2) around the model, the strict ensembles' truths)
    ln p_chain(S)     = -(k/2 - 1) ln S                              (flat chain prior, uniform in C)
                      = -(k+nu)/2 ln(S + nu C_chain^fid)              (nu chain prior)
up to theta-independent constants. The product over l is exact on the full sky with
isotropic noise (the multipoles are independent) and an approximation under a mask; the
joint (unfactorised) BR over l = 2-59 is unusable here: its importance weights collapse
to an ESS of ~2 of 3600 samples (job 12200810), because each multipole's log-weight
spreads by ~sqrt(2 N_l / C_l) and 58 of them add. Per multipole the ESS is reported.

The model is C_l(A_30, n_s) = A_30 x [r_l(n_s) / r_30(n_s)] x the chain's fiducial, with
r_l(n_s) = CAMB(n_s) / CAMB(0.965) and the other parameters at diffcmb.power.LCDM_PARAMS
(tau = 0.066): A_30 = C_30 / C_30^fid is the amplitude at the pivot l = 30, where these
multipoles decorrelate it from n_s. (Earlier grids truncated the posteriors along the
A-n_s ridge: A = A_s / A_s^fid to 1.4, job 12201039, and A_80 to 1.3, job 12201134, both
with up to the full peak density on the boundary. Each run now reports the boundary
density and warns above EDGE_WARN.) Flat priors on (A_30, n_s). The band edge (l >= 60) is excluded by
default, as in every pooled number of the paper.

Modes:
  --chains DIR [DIR ...]   one run (several chains) or several noise draws of one map
                           (each draw's L normalised and averaged: the noise-marginalised
                           posterior, as realdata_summary.py pools them);
  --ensemble DIR           every chain_rNNN.npz of a simulation ensemble, one posterior per
                           sky, and the posterior CDF at the truth (A = 1, n_s = 0.965):
                           --target lcdm on a flat-prior ensemble (every sky's spectrum is the
                           fiducial), --target hier on a strict one (truths drawn from the nu
                           prior around it) -- uniform if the likelihood is calibrated.

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/br_lcdm.py --ensemble \\
      results/analysis/ens_exact_l64_A3000_n30_nu30_long --out results/analysis/br_lcdm/d5
"""
import argparse
import glob
import os

import numpy as np
from scipy.special import logsumexp

from diffcmb.alm_utils import packed_sizes
from diffcmb.power import LCDM_PARAMS

NS_FID = 0.965
LMIN, LMAX_USE = 2, 60  # l in [LMIN, LMAX_USE): the band edge l >= 60 is left out
L_PIVOT = 30  # A_30 = C_30 / C_30^fid; the A-n_s ridge has d ln A_80 / d n_s ~ 1, so ~80/e decorrelates
A_GRID = np.linspace(0.6, 1.5, 361)
NS_GRID = np.linspace(0.3, 1.7, 141)
EDGE_WARN = 1e-3  # boundary density / peak above this: the grid truncates the posterior


def sl_weights(lmax):
    """(P, lmax) matrix W with S = a**2 @ W: the packed layout of model.compute_sl_np
    (real parts, l = 2..lmax-1, m = 0..l, weight 1 at m = 0 and 2 above; then imaginary
    parts, m = 1..l, weight 2)."""
    n_real, n_imag = packed_sizes(lmax)
    w = np.zeros((n_real + n_imag, lmax))
    r, i = 0, n_real
    for ell in range(2, lmax):
        w[r, ell] = 1.0
        w[r + 1:r + ell + 1, ell] = 2.0
        w[i:i + ell, ell] = 2.0
        r += ell + 1
        i += ell
    return w


def sl_from_packed(alm, lmax):
    """S_l, l = 0..lmax-1, for each row of (n, P) packed a_lm."""
    return np.asarray(alm) ** 2 @ sl_weights(lmax)


def dof(lmax):
    """Real dof per multipole, l = 0..lmax-1 (2l+1 from l = 2)."""
    k = 2.0 * np.arange(lmax) + 1
    k[:2] = 0
    return k


def chain_log_marginal(s, k, nu=None, cl_fid=None):
    """ln p_chain(S_l) per sample and multipole, (n, n_l), up to constants: the chain's
    C_l prior integrated out (flat if nu is None, else InvGamma(nu/2, nu cl_fid/2))."""
    if nu is None:
        return -(k / 2 - 1) * np.log(s)
    return -(k + nu) / 2 * np.log(s + nu * cl_fid)


def target_log_p(s, k, c, target="lcdm", target_nu=None):
    """ln p_target(S | c) for (n, 1) S against (1, n_c) model values c, up to constants."""
    if target == "lcdm":
        return -k / 2 * np.log(c) - s / (2 * c)
    if target == "hier":
        return target_nu / 2 * np.log(c) - (k + target_nu) / 2 * np.log(s + target_nu * c)
    raise ValueError(f"unknown target {target!r}")


def ell_log_like(s_l, k_l, c_vals, log_marg_l, target="lcdm", target_nu=None):
    """One multipole: ln L_l(c) and the importance ESS for each model value c (n_c,)."""
    logw = target_log_p(s_l[:, None], k_l, c_vals[None], target, target_nu) - log_marg_l[:, None]
    ll = logsumexp(logw, axis=0) - np.log(s_l.size)
    w = np.exp(logw - logw.max(0))
    return ll, w.sum(0) ** 2 / (w ** 2).sum(0)


def log_like_grid(s, k, cl_grid, log_marg, target="lcdm", target_nu=None, n_c=801):
    """ln L(theta) = sum_l ln L_l(C_l(theta)) for each row of cl_grid ((n_theta, n_l)), and
    each theta's smallest per-multipole importance ESS. Each L_l is tabulated on n_c values
    of ln C spanning that multipole's column of cl_grid and interpolated."""
    ll = np.zeros(cl_grid.shape[0])
    ess = np.full(cl_grid.shape[0], np.inf)
    for j in range(cl_grid.shape[1]):
        lc = np.log(cl_grid[:, j])
        if np.ptp(lc) == 0:
            lj, ej = ell_log_like(s[:, j], k[j], cl_grid[:1, j], log_marg[:, j], target, target_nu)
            ll += lj[0]
            ess = np.minimum(ess, ej[0])
            continue
        table = np.linspace(lc.min(), lc.max(), n_c)
        lj, ej = ell_log_like(s[:, j], k[j], np.exp(table), log_marg[:, j], target, target_nu)
        ll += np.interp(lc, table, lj)
        ess = np.minimum(ess, np.interp(lc, table, ej))
    return ll, ess


def tilt_ratio(lmax, ns_grid):
    """CAMB unlensed C_l^TT(n_s) / C_l^TT(NS_FID), (len(ns_grid), max(lmax, L_PIVOT + 1)),
    l < 2 set to 1."""
    import camb

    p = LCDM_PARAMS

    def tt(ns):
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=p[0], ombh2=p[1], omch2=p[2], mnu=p[3], omk=p[4], tau=p[5])
        pars.InitPower.set_params(As=2e-9, ns=ns, r=0)
        pars.set_for_lmax(max(n + 100, 500), lens_potential_accuracy=1)
        res = camb.get_results(pars)
        return res.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)["unlensed_scalar"][:n, 0]

    n = max(lmax, L_PIVOT + 1)
    ref = tt(NS_FID)
    out = np.ones((len(ns_grid), n))
    for j, ns in enumerate(ns_grid):
        out[j, 2:] = tt(ns)[2:] / ref[2:]
    return out


def model_grid(cl_base, ratio):
    """(n_A * n_ns, n_l) model spectra on l in [LMIN, LMAX_USE) from tilt_ratio's full output,
    n_s fastest: index = iA * n_ns + jns. Normalised to A_80 at L_PIVOT."""
    shape = ratio[:, LMIN:LMAX_USE] / ratio[:, L_PIVOT:L_PIVOT + 1]
    return (A_GRID[:, None, None] * shape[None] * cl_base[None, None]).reshape(-1, cl_base.size)


def edge_fraction(p):
    """Largest posterior density on the grid boundary over the peak density."""
    return max(p[0].max(), p[-1].max(), p[:, 0].max(), p[:, -1].max()) / p.max()


def chain_s(path, lmax_use=LMAX_USE):
    """S_l over l in [LMIN, lmax_use) for every sample of one chain, and its prior."""
    d = np.load(path)
    lmax = int(d["lmax"])
    s = sl_from_packed(d["alm_samples"][:, lmax - 2:], lmax)[:, LMIN:lmax_use]
    nu = float(d["cl_prior_nu"]) if "cl_prior_nu" in d.files and np.isfinite(d["cl_prior_nu"]) else None
    fid = (d["cl_tt_fid"] if "cl_tt_fid" in d.files else d["cl_true"])[LMIN:lmax_use]
    return s, nu, np.asarray(fid, dtype=np.float64), lmax


def posterior(ll):
    """Normalised posterior on the (A, n_s) grid (flat priors) from ln L."""
    p = np.exp(ll - ll.max()).reshape(A_GRID.size, NS_GRID.size)
    return p / p.sum()


def summarise(p):
    """Marginal means and sds of A and n_s."""
    pa, pn = p.sum(1), p.sum(0)
    ma, mn = (pa * A_GRID).sum(), (pn * NS_GRID).sum()
    return {"A": ma, "A_sd": np.sqrt((pa * (A_GRID - ma) ** 2).sum()),
            "ns": mn, "ns_sd": np.sqrt((pn * (NS_GRID - mn) ** 2).sum())}


def cdf_at_truth(p):
    """Posterior mass below the truth (A_pivot = 1, n_s = NS_FID), half the truth's cell included."""
    pa, pn = p.sum(1), p.sum(0)
    return (pa[A_GRID < 1.0].sum() + 0.5 * pa[np.isclose(A_GRID, 1.0)].sum(),
            pn[NS_GRID < NS_FID].sum() + 0.5 * pn[np.isclose(NS_GRID, NS_FID)].sum())


def sky_row(s, k, grid, log_marg, target, target_nu):
    """(A, A_sd, n_s, n_s_sd, CDF_A, CDF_ns, min ESS, edge) of one sky's posterior."""
    ll, ess = log_like_grid(s, k, grid, log_marg, target, target_nu)
    p = posterior(ll)
    t = summarise(p)
    return (t["A"], t["A_sd"], t["ns"], t["ns_sd"], *cdf_at_truth(p),
            ess.reshape(p.shape)[p > 1e-3 * p.max()].min(), edge_fraction(p))


def report_calibration(r, label):
    """Mean/sd z, rank-based z (Phi^-1 of the CDF at the truth: no Gaussian assumption) and KS."""
    from scipy.stats import kstest, norm

    za, zn = (r[:, 0] - 1.0) / r[:, 1], (r[:, 2] - NS_FID) / r[:, 3]
    ra, rn = (norm.ppf(1 - np.clip(r[:, c], 1e-9, 1 - 1e-9)) for c in (4, 5))
    print(f"{label}, {len(r)} skies: z(A_{L_PIVOT}) mean {za.mean():+.2f} sd {za.std(ddof=1):.2f}; "
          f"z(n_s) mean {zn.mean():+.2f} sd {zn.std(ddof=1):.2f}; rank-z sd {ra.std(ddof=1):.2f} / "
          f"{rn.std(ddof=1):.2f}; KS p {kstest(r[:, 4], 'uniform').pvalue:.2f} / "
          f"{kstest(r[:, 5], 'uniform').pvalue:.2f}")


def run_draws(dirs, ratio_cache, target="lcdm"):
    """Noise-marginalised posterior over several runs (each a list of chains)."""
    posts, ess_min = [], []
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "data_chain_r[0-9][0-9][0-9].npz")))
        if not files:
            raise SystemExit(f"no data_chain_rNNN.npz in {d}")
        parts = [chain_s(f) for f in files]
        s = np.concatenate([x[0] for x in parts])
        nu, fid, lmax = parts[0][1], parts[0][2], parts[0][3]
        k = dof(lmax)[LMIN:LMAX_USE]
        grid = model_grid(fid, ratio_cache(lmax))
        ll, ess = log_like_grid(s, k, grid, chain_log_marginal(s, k, nu=nu, cl_fid=fid),
                                target, nu if target == "hier" else None)
        p = posterior(ll)
        posts.append(p)
        ess_min.append(ess.reshape(p.shape)[p > 1e-3 * p.max()].min())
    return np.mean(posts, axis=0), np.array(ess_min), posts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--chains", nargs="+")
    g.add_argument("--ensemble")
    ap.add_argument("--target", choices=("lcdm", "hier"), default="lcdm",
                    help="lcdm: C_l = C_l(theta) exactly; hier: C_l ~ InvGamma(nu/2, nu C_l(theta)/2) "
                         "with the chain's nu (the strict ensembles' truths)")
    ap.add_argument("--out", required=True, help="output prefix (.npz written)")
    a = ap.parse_args()
    cache = {}

    def ratio_cache(lmax):
        if lmax not in cache:
            cache[lmax] = tilt_ratio(lmax, NS_GRID)
        return cache[lmax]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print(f"l in [{LMIN},{LMAX_USE}); grid A_{L_PIVOT} {A_GRID[0]}-{A_GRID[-1]} ({A_GRID.size}), "
          f"n_s {NS_GRID[0]}-{NS_GRID[-1]} ({NS_GRID.size}); flat priors on the grid")
    if a.chains:
        p, ess_min, per = run_draws(a.chains, ratio_cache, a.target)
        s = summarise(p)
        if edge_fraction(p) > EDGE_WARN:
            print(f"!! posterior truncated: boundary density {edge_fraction(p):.2g} of the peak; widen the grid")
        print(f"{len(a.chains)} run(s): A_{L_PIVOT} = {s['A']:.3f} +- {s['A_sd']:.3f}, "
              f"n_s = {s['ns']:.3f} +- {s['ns_sd']:.3f}; min importance ESS in the posterior "
              f"bulk per run {np.array2string(ess_min, precision=0)}")
        for d, q in zip(a.chains, per):
            t = summarise(q)
            print(f"  {d}: A_{L_PIVOT} = {t['A']:.3f} +- {t['A_sd']:.3f}, n_s = {t['ns']:.3f} +- {t['ns_sd']:.3f}")
        np.savez(a.out + ".npz", post=p, per_draw=np.array(per), ess_min=ess_min, A_grid=A_GRID,
                 ns_grid=NS_GRID, chains=np.array(a.chains), lmin=LMIN, lmax_use=LMAX_USE,
                 target=a.target, **{f"mean_{k}": v for k, v in s.items()})
    else:
        files = sorted(glob.glob(os.path.join(a.ensemble, "chain_r[0-9][0-9][0-9].npz")))
        rows, oracle = [], []
        for f in files:
            sv, nu, fid, lmax = chain_s(f)
            d = np.load(f)
            truth = d["cl_true"][LMIN:LMAX_USE]
            if a.target == "lcdm" and not np.allclose(truth, fid, rtol=1e-10):
                raise SystemExit(f"{f}: the truth is not the fiducial spectrum, so (A, n_s) = "
                                 "(1, 0.965) is not its truth under --target lcdm; a strict "
                                 "ensemble (truths drawn from the nu prior) needs --target hier")
            if a.target == "hier" and nu is None:
                raise SystemExit(f"{f}: --target hier needs a chain with a nu prior on C_l")
            k = dof(lmax)[LMIN:LMAX_USE]
            grid = model_grid(fid, ratio_cache(lmax))
            tnu = nu if a.target == "hier" else None
            rows.append(sky_row(sv, k, grid, chain_log_marginal(sv, k, nu=nu, cl_fid=fid), a.target, tnu))
            # the oracle: the true unlensed a_lm as the only "sample" (no noise, no lensing, full sky)
            st = sl_from_packed(d["alm_true_packed"][None], lmax)[:, LMIN:LMAX_USE]
            oracle.append(sky_row(st, k, grid, chain_log_marginal(st, k), a.target, tnu))
            print(f"  {os.path.basename(f)}: A_{L_PIVOT} {rows[-1][0]:.3f} +- {rows[-1][1]:.3f}  n_s "
                  f"{rows[-1][2]:.3f} +- {rows[-1][3]:.3f}  CDF(truth) {rows[-1][4]:.2f} / {rows[-1][5]:.2f}  "
                  f"min ESS {rows[-1][6]:.0f}  | oracle A {oracle[-1][0]:.3f} n_s {oracle[-1][2]:.3f}")
        r, o = np.array(rows), np.array(oracle)
        if r[:, 7].max() > EDGE_WARN or o[:, 7].max() > EDGE_WARN:
            print(f"!! posteriors truncated by the grid (max boundary density {max(r[:, 7].max(), o[:, 7].max()):.2g} "
                  "of the peak); widen the grid")
        print()
        report_calibration(r, "chains")
        report_calibration(o, "oracle (true a_lm)")
        for lab, c in (("A", 0), ("n_s", 2)):
            dz = (r[:, c] - o[:, c]) / o[:, c + 1]
            ratio_sd = np.median(r[:, c + 1] / o[:, c + 1])
            print(f"chain - oracle, {lab}: mean {dz.mean():+.2f}, rms {np.sqrt(np.mean(dz ** 2)):.2f} oracle sd "
                  f"(a correct posterior: ~{np.sqrt(max(ratio_sd ** 2 - 1, 0)):.2f}); sd ratio {ratio_sd:.2f}, "
                  f"corr {np.corrcoef(r[:, c], o[:, c])[0, 1]:.3f}")
        print(f"median min-ESS {np.median(r[:, 6]):.0f}")
        np.savez(a.out + ".npz", rows=r, oracle=o, files=np.array(files), A_grid=A_GRID, ns_grid=NS_GRID,
                 lmin=LMIN, lmax_use=LMAX_USE, l_pivot=L_PIVOT, target=a.target)
    print(f"saved {a.out}.npz")


if __name__ == "__main__":
    main()
