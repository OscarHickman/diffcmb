"""Null calibration for the coverage-ensemble rank statistics (2026-08-31).

WHY THIS EXISTS. `scripts/calibration/aggregate_coverage_ranks.py` FLAGs a bin when its
rank distribution is non-uniform. For the *spectrum* rows (C_l^TT, C_L^phiphi)
that flag fires even for a perfect sampler, because the statistic ranks the
truth against its own conditional's MODE:

    Blocks 1 and 4 both draw   C ~ InvGamma(alpha = k_L/2 + a0, beta = (S_L + b0_L)/2)
    whose mode under the flat  beta/(alpha+1) = S_L / k_L
    prior (a0 = -1, b0 = 0) is
    which is exactly           aggregate_coverage_ranks.realized_spectrum(S)

    k_L = 2L is the PACKED real dof (splittosingularalm forces Im(a_{L,1}) = 0).
    Until 2026-08-31 this script hardcoded alpha = L - 0.5 (i.e. k_L = 2L+1) in
    all three checks while the sampler had already been fixed to derive alpha
    from the packing, so its null band and its Block-4 CDF were computed from
    the wrong distribution. Both now read the shape AND the prior off the saved
    chain, so the null tracks the configuration it is validating.

InvGamma is right-skewed, so P(draw < mode) < 0.5 by construction, and
bin-averaging over l shrinks the spread while leaving the offset -- driving
mean_u toward 0. Reading those flags as sampler bias is a mistake; this script
computes what a correct sampler actually produces so observed can be compared
against the right null.

It runs three checks:

  1. NULL, conditioning variable frozen at truth. Cheap, and adequate for
     C_l^TT because alm is pinned tightly by the data (cosine ~0.9998).
  2. NULL, using the chain's OWN phi trajectory. Required for C_L^phiphi:
     phi moves sweep-to-sweep, adding common-mode spread across l that does
     NOT average down within a bin, so the frozen-phi null is far too narrow
     and makes a correct sampler look biased.
  3. BLOCK 4 EXACTNESS, directly on production chains. Using the saved
     per-sweep (phi_samples, cl_phiphi_samples) pairs,
     u = CDF_InvGamma(k_L/2+a0, (S_L(phi_i)+b0_L)/2)[C_{L,i}] must be
     Uniform(0,1). A lag-1 misalignment is run as a control so a pass can't
     be vacuous.

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/calibration/validate_coverage_rank_nulls.py \
      --indir results/analysis/coverage_ensemble_lmax64_prior_cl4_mapfix --thin 90
"""
import argparse
import glob
import os

import numpy as np
from aggregate_coverage_ranks import _sl_from_packed, realized_spectrum
from scipy import stats
from scipy.special import gammaincc

from diffcmb.alm_utils import invgamma_shape_for_spectrum
from diffcmb.lensing import compute_sl_phi_np

BINS = [(2, 10), (10, 30), (30, 60), (60, 64)]


def prior_of(d, tag):
    """(a0, b0_vec) of the InvGamma prior the run actually used, from the chain.

    Block 1 (C_l^TT) is always the flat improper default, a0 = -1, b0 = 0.
    Block 4 (C_L^phiphi) is flat too unless the run set --cl_phiphi_prior_nu,
    in which case the proper conjugate prior is InvGamma(nu/2, nu*C_L^fid/2)
    and BOTH the shape and the rate pick up a prior term. Read from the saved
    chain rather than assumed, so this script cannot silently drift from the
    configuration it is validating (as it did before 2026-08-31, when it
    hardcoded alpha = L - 0.5 for every run).
    """
    lmax = int(d["lmax"])
    if tag == "cl_phiphi":
        nu = float(d["cl_phiphi_prior_nu"]) if "cl_phiphi_prior_nu" in d.files else 0.0
        if np.isfinite(nu) and nu > 0.0:
            return 0.5 * nu, nu * np.asarray(d["cl_phiphi_fid"], dtype=np.float64)[:lmax]
    return -1.0, np.zeros(lmax, dtype=np.float64)


def alpha_of(d, tag):
    """Per-multipole InvGamma shape the run's conditional actually used."""
    a0, _ = prior_of(d, tag)
    return invgamma_shape_for_spectrum(int(d["lmax"]), a0=a0)


def load(indir):
    files = sorted(
        f for f in glob.glob(os.path.join(indir, "chain_r*.npz"))
        if not f.endswith("_ckpt.npz")
    )
    if not files:
        raise SystemExit(f"no chains found in {indir}")
    return files


def _rank_u(post, tru, n_eff):
    return (int(np.sum(post < tru)) + 0.5) / (n_eff + 1.0)


def null_frozen(files, tag, n_eff, n_rep, rng):
    """Null with the conditioning field frozen at truth."""
    per_real = []
    for f in files:
        d = np.load(f)
        lmax = int(d["lmax"])
        S = (_sl_from_packed(d["alm_true_packed"], lmax) if tag == "cl_TT"
             else compute_sl_phi_np(d["phi_true_packed"], lmax))
        _a0, b0 = prior_of(d, tag)
        per_real.append((S, lmax, alpha_of(d, tag), b0))

    print(f"\n--- NULL [{tag}]: conditioning field FROZEN at truth ---")
    print(f"  {'bin':12s} {'null_mean':>10s} {'2.5%':>9s} {'97.5%':>9s}")
    for lo, hi in BINS:
        means = np.empty(n_rep)
        for r in range(n_rep):
            us = []
            for S, lmax, alpha, b0 in per_real:
                ells = np.arange(max(2, lo), min(lmax, hi))
                beta = (S[ells] + b0[ells]) / 2.0
                draws = beta[None, :] / rng.gamma(
                    alpha[ells][None, :], 1.0, size=(n_eff, len(ells))
                )
                us.append(_rank_u(draws.mean(axis=1),
                                  realized_spectrum(S, lmax)[ells].mean(), n_eff))
            means[r] = np.mean(us)
        print(f"  [{lo:3d},{hi:3d})   {means.mean():10.4f} "
              f"{np.percentile(means, 2.5):9.4f} {np.percentile(means, 97.5):9.4f}")


def null_with_phi_trajectory(files, thin, n_rep, rng):
    """Null for C_L^phiphi that keeps the chain's real sweep-to-sweep phi scatter."""
    per_real = []
    for f in files:
        d = np.load(f)
        lmax = int(d["lmax"])
        S_chain = np.array([compute_sl_phi_np(p, lmax)
                            for p in d["phi_samples"][::thin]])
        _a0, b0 = prior_of(d, "cl_phiphi")
        per_real.append((S_chain, compute_sl_phi_np(d["phi_true_packed"], lmax),
                         lmax, alpha_of(d, "cl_phiphi"), b0))
    n_eff = per_real[0][0].shape[0]

    print(f"\n--- NULL [cl_phiphi]: using the chain's OWN phi trajectory "
          f"(n_eff={n_eff}) ---")
    print(f"  {'bin':12s} {'null_mean':>10s} {'2.5%':>9s} {'97.5%':>9s}")
    for lo, hi in BINS:
        means = np.empty(n_rep)
        for r in range(n_rep):
            us = []
            for S_chain, S_true, lmax, alpha, b0 in per_real:
                ells = np.arange(max(2, lo), min(lmax, hi))
                beta = (S_chain[:, ells] + b0[ells][None, :]) / 2.0
                draws = beta / rng.gamma(alpha[ells][None, :], 1.0, size=beta.shape)
                us.append(_rank_u(draws.mean(axis=1),
                                  realized_spectrum(S_true, lmax)[ells].mean(), n_eff))
            means[r] = np.mean(us)
        print(f"  [{lo:3d},{hi:3d})   {means.mean():10.4f} "
              f"{np.percentile(means, 2.5):9.4f} {np.percentile(means, 97.5):9.4f}")


def _block4_u(files, stride, lag):
    """PIT values u = F(C_L | S_L(phi_{i-lag})) over every chain, lag-shifted."""
    allu = []
    for f in files:
        d = np.load(f)
        lmax = int(d["lmax"])
        phi, lnC = d["phi_samples"], d["cl_phiphi_samples"]
        if lag:
            phi, lnC = phi[:-lag], lnC[lag:]
        ells = np.arange(2, lmax)
        alpha = alpha_of(d, "cl_phiphi")[ells]
        _a0, b0 = prior_of(d, "cl_phiphi")
        for i in range(0, len(phi), stride):
            S = compute_sl_phi_np(phi[i], lmax)[ells]
            allu.append(gammaincc(alpha, ((S + b0[ells]) / 2.0) / np.exp(lnC[i])))
    return np.concatenate(allu)


def block4_exactness(files, stride, control_lags):
    """Is Block 4 drawing from its own exact conditional given its own phi?

    The conditional is InvGamma(k_L/2 + a0, (S_L(phi) + b0_L)/2) with k_L = 2L
    the packed dof and (a0, b0) the run's prior -- flat (-1, 0) by default, or
    (nu/2, nu*C_L^fid) when --cl_phiphi_prior_nu was set. Both are read from
    the chain, so this check follows whatever configuration produced it.

    The aligned statistic is only evidence if a deliberately MISALIGNED pairing
    is rejected, and lag-1 alone is not enough: it detects the misalignment only
    when phi actually moves between consecutive sweeps. On job 11903182 (Block 4
    ON, proper prior) the lag-1 control passed at KS_p=0.183 because
    S_L(phi_i) ~= S_L(phi_{i+1}) under the Block-4-ON funnel, which made the
    aligned pass vacuous. Several lags are therefore run, spaced out to the
    scale on which phi actually decorrelates, and the verdict below states
    explicitly whether any control had power.
    """
    print("\n--- BLOCK 4 EXACTNESS on production chains ---")
    print("  u = CDF_InvGamma(k_L/2+a0, (S_L(phi_i)+b0_L)/2)[C_{L,i}]; "
          "uniform if exact.")
    if not all("cl_phiphi_samples" in np.load(f).files for f in files):
        print("  (Block 4 disabled in these chains -- nothing to check; skipped)")
        return
    lags = [(0, "aligned")] + [(k, f"lag-{k} (control, should FAIL)")
                               for k in control_lags]
    pvals = {}
    for lag, label in lags:
        u = _block4_u(files, stride, lag)
        ks = stats.kstest(u, "uniform")
        pvals[lag] = ks.pvalue
        print(f"  {label:30s} N={u.size:6d}  mean_u={u.mean():.4f}  "
              f"KS_D={ks.statistic:.4f}  KS_p={ks.pvalue:.3g}")

    _report_phi_decorrelation(files, control_lags)

    powered = [k for k in control_lags if pvals[k] < 1e-3]
    if not powered:
        print("  VERDICT: NO control was rejected at p<1e-3 -- phi barely moves "
              "on any lag tried, so\n           the aligned result is VACUOUS. "
              "Do not report it as an exactness pass;\n           raise "
              "--control_lags, or report it only with a control that fails.")
    else:
        verdict = "PASS" if pvals[0] > 0.01 else "FAIL"
        print(f"  VERDICT: controls at lag {powered} are rejected, so the test "
              f"has power;\n           the aligned statistic is a genuine "
              f"{verdict} (KS_p={pvals[0]:.3g}).")


def _report_phi_decorrelation(files, control_lags):
    """How far apart must two sweeps be before S_L(phi) actually differs?

    This is what sets whether a lag-k control can have power at all: the PIT
    can only notice a misalignment that changes its conditioning variable.
    """
    print("  phi decorrelation (mean over L of corr[S_L(phi_i), S_L(phi_{i+k})]):")
    for k in control_lags:
        cs = []
        for f in files:
            d = np.load(f)
            lmax = int(d["lmax"])
            S = np.array([compute_sl_phi_np(p, lmax)[2:lmax]
                          for p in d["phi_samples"]])
            if len(S) <= k:
                continue
            a, b = S[:-k], S[k:]
            a = a - a.mean(axis=0)
            b = b - b.mean(axis=0)
            denom = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
            cs.append(np.mean((a * b).sum(axis=0) / np.where(denom > 0, denom, 1)))
        if cs:
            print(f"    lag {k:4d}: {np.mean(cs):+.3f}")


def strict_clpp_sbc(files, thin):
    """STRICT SBC rank of C_L^phiphi_true among the Block 4 samples.

    Only defined when the run used a PROPER prior (--cl_phiphi_prior_nu), because
    only then was the truth drawn from the same joint prior the sampler targets
    (coverage_ensemble_chain.py draws C_L ~ InvGamma, then phi ~ N(0,C_L)). Under
    a correct sampler this rank is uniform -- no null simulation needed, unlike
    the aggregator's C_L^phiphi row, which ranks against the REALIZED power of
    phi_true and is therefore coverage only.

    This is the statistic that exposed the 2026-08-31 inverse-Gamma dof bug
    (mean_u = 0.25-0.28, KS_p = 0.0000 with alpha = L - 0.5), and it is the
    decisive check that the corrected alpha = k_L/2 + a0 fixed it.
    """
    print("\n--- STRICT C_L^phiphi SBC rank (truth drawn from the sampler's own "
          "prior) ---")
    d0 = np.load(files[0])
    nu = float(d0["cl_phiphi_prior_nu"]) if "cl_phiphi_prior_nu" in d0.files else 0.0
    if "cl_phiphi_samples" not in d0.files or not (np.isfinite(nu) and nu > 0.0):
        print("  (needs Block 4 ON with --cl_phiphi_prior_nu; skipped)")
        return
    print(f"  prior nu={nu:g}; uniform under a correct sampler, no null needed.")
    print(f"  {'bin':12s} {'N':>4s} {'mean_u':>8s} {'KS_p':>8s}   ranks")
    allu = []
    for lo, hi in BINS:
        us, ranks = [], []
        for f in files:
            d = np.load(f)
            lmax = int(d["lmax"])
            ells = np.arange(max(2, lo), min(lmax, hi))
            # cl_phiphi_samples is stored as log C_L for l=2..lmax-1.
            post = np.exp(d["cl_phiphi_samples"][::thin])[:, ells - 2].mean(axis=1)
            tru = np.asarray(d["cl_phiphi_true"], dtype=np.float64)[ells].mean()
            ranks.append(int(np.sum(post < tru)))
            us.append((ranks[-1] + 0.5) / (len(post) + 1.0))
        allu.extend(us)
        ks = stats.kstest(np.asarray(us), "uniform").pvalue if len(us) >= 5 else np.nan
        flag = "  <-- FLAG" if ks < 0.01 else ""
        print(f"  [{lo:3d},{hi:3d})  {len(us):4d} {np.mean(us):8.4f} {ks:8.4f}   "
              f"{' '.join(str(r) for r in ranks)}{flag}")
    u = np.asarray(allu)
    print(f"  POOLED: N={u.size}  mean_u={u.mean():.4f}  "
          f"KS_p={stats.kstest(u, 'uniform').pvalue:.4g}")


def phi_power_bias(files, stride):
    """Per-bin <S_L(phi_chain)> / S_L(phi_true): is phi itself over-powered?"""
    print("\n--- phi power bias, per bin (>1 = posterior phi over-powered) ---")
    tab = {b: [] for b in BINS}
    for f in files:
        d = np.load(f)
        lmax = int(d["lmax"])
        S_true = compute_sl_phi_np(d["phi_true_packed"], lmax)
        phi = d["phi_samples"]
        S_chain = np.mean([compute_sl_phi_np(phi[i], lmax)
                           for i in range(0, len(phi), stride)], axis=0)
        for lo, hi in BINS:
            ells = np.arange(lo, min(hi, lmax))
            tab[(lo, hi)].append(S_chain[ells].sum() / S_true[ells].sum())
    print(f"  {'bin':12s} {'median':>8s} {'min':>8s} {'max':>8s}")
    for b in BINS:
        v = np.array(tab[b])
        print(f"  [{b[0]:3d},{b[1]:3d})  {np.median(v):8.3f} {v.min():8.3f} "
              f"{v.max():8.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--indir", required=True)
    p.add_argument("--thin", type=int, default=90,
                   help="must match the --thin passed to aggregate_coverage_ranks.py")
    p.add_argument("--n_rep", type=int, default=400, help="null replications")
    p.add_argument("--stride", type=int, default=25,
                   help="sweep subsampling for the Block 4 / phi-power checks")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--control_lags", type=str, default="1,10,50",
                   help="comma-separated lags for the Block 4 PIT misalignment "
                        "controls; at least one must be rejected or the aligned "
                        "pass is vacuous")
    args = p.parse_args()

    files = load(args.indir)
    rng = np.random.default_rng(args.seed)
    n_eff = len(np.load(files[0])["phi_samples"][::args.thin])
    print(f"=== Rank-null calibration: {len(files)} realizations, "
          f"thin={args.thin} (n_eff={n_eff}) ===")

    for tag in ("cl_TT", "cl_phiphi"):
        null_frozen(files, tag, n_eff, args.n_rep, rng)
    null_with_phi_trajectory(files, args.thin, args.n_rep, rng)
    block4_exactness(files, args.stride,
                     [int(k) for k in args.control_lags.split(',') if k])
    strict_clpp_sbc(files, args.thin)
    phi_power_bias(files, args.stride)

    print("\nCompare each observed mean_u from aggregate_coverage_ranks.py against\n"
          "the matching null band above -- NOT against 0.5. For C_L^phiphi use the\n"
          "phi-trajectory null; the frozen-phi one is too narrow.")


if __name__ == "__main__":
    main()
