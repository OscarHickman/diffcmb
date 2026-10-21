"""Re-score saved phi-block pilot chains with stationarity/ESS diagnostics.

Motivation (2026-08-26). The standing equilibration gate is "worst lag-1
autocorrelation < 0.9". That single number conflates two physically distinct
failure modes:

  (a) SLOW BUT STATIONARY -- high r_1, but the autocorrelation decays to zero
      within the window and there is no drift. This is an ordinary slowly-mixing
      chain. It is scientifically usable: SBC/coverage ranks need stationarity
      and an honest effective sample size, NOT a short correlation time.

  (b) NOT EQUILIBRATED -- a long-lived mode that survives to large lag, usually
      with drift. A rank test built on this produces confidently wrong plots.

Raw lag-1 fails both identically, so every pilot has been scored NO-GO without
distinguishing them. This script separates them, using only chains already on
disk (no new sampling).

Diagnostics per l-bin:
  * tau_int  -- Geyer initial-positive-sequence integrated autocorrelation time.
  * ESS      -- N / tau_int, the number of effectively independent draws.
  * split-Rhat -- standard stationarity diagnostic (split the chain in half).
  * AR1 ratio -- tau_int divided by the AR(1) time implied by the SAME lag-1,
      tau_AR1 = (1+r_1)/(1-r_1). A chain whose autocorrelation really is a
      single exponential has ratio ~1; a chain carrying an extra long-lived
      mode has ratio >> 1. This is the discriminator between (a) and (b), and
      it is the thing raw lag-1 cannot see: two bins with identical r_1 can
      have very different tails.

      (Deliberately NOT r_k / r_1^k at fixed k: for small r_1 the denominator
      underflows and the ratio explodes to meaningless 1e11 values. The
      tau ratio is well-conditioned across the whole r_1 range.)
"""

import argparse
import os

import numpy as np


def autocorr(x, max_lag):
    """Normalised autocorrelation function r_k for k = 0..max_lag."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    var = np.dot(x, x)
    if var <= 0:
        return np.full(max_lag + 1, np.nan)
    n = len(x)
    return np.array([np.dot(x[: n - k], x[k:]) / var for k in range(max_lag + 1)])


def tau_int_geyer(r):
    """Integrated autocorrelation time via Geyer's initial positive sequence.

    Sums adjacent pairs Gamma_m = r_{2m} + r_{2m+1} and truncates at the first
    non-positive pair -- the standard self-consistent truncation that avoids the
    noise blow-up of a naive full sum.
    """
    gammas = []
    for m in range(len(r) // 2):
        g = r[2 * m] + r[2 * m + 1]
        if g <= 0:
            break
        gammas.append(g)
    if not gammas:
        return 1.0, True
    # Ran out of lags without ever finding a non-positive pair => the
    # autocorrelation hadn't decayed within the window, so the sum is a lower
    # bound only, not a converged estimate.
    truncated_early = len(gammas) < len(r) // 2
    return max(1.0, 2.0 * sum(gammas) - 1.0), truncated_early


def split_rhat(x):
    """Split-Rhat on a single chain (split in half, treat as two chains)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x) // 2
    if n < 2:
        return np.nan
    chains = np.array([x[:n], x[n : 2 * n]])
    means, variances = chains.mean(axis=1), chains.var(axis=1, ddof=1)
    W = variances.mean()
    if W <= 0:
        return np.nan
    B = n * means.var(ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


# Gate thresholds. ESS_MIN is per-chain: with an O(10-20)-chain coverage
# ensemble, 20 effective draws per chain pools to several hundred, which is a
# defensible rank test.
ESS_MIN = 20.0
RHAT_MAX = 1.05
DRIFT_MAX = 2.0
AR1_EXCESS_MAX = 3.0


def score_file(path):
    d = np.load(path, allow_pickle=True)
    traces = d["phi_traces"]
    bins = d["phi_trace_bins"]
    n = traces.shape[1]
    max_lag = min(400, n // 3)

    print(f"\n{'=' * 78}\n{os.path.basename(path)}")
    print(
        f"  lmax={int(d['lmax'])} nside={int(d['nside'])} "
        f"phi_n_lfs={int(d['phi_n_lfs'])} N={n}"
    )
    print(
        f"\n  {'l-bin':>12} {'r_1':>7} {'tau_int':>8} {'ESS':>7} "
        f"{'Rhat':>6} {'drift':>7} {'AR1rat':>7}  verdict"
    )

    rows = []
    for i, (lo, hi) in enumerate(bins):
        t = traces[i]
        r = autocorr(t, max_lag)
        tau, converged = tau_int_geyer(r)
        ess = n / tau
        rhat = split_rhat(t)

        # Drift: (mean of last third - mean of first third) in units of sd.
        third = n // 3
        sd = t.std(ddof=1)
        drift = (t[-third:].mean() - t[:third].mean()) / sd if sd > 0 else np.nan

        # AR(1) ratio: tau_int against the AR(1) time implied by the same
        # lag-1. ~1 => the decay really is a single exponential (ordinary slow
        # mixing); >> 1 => an extra long-lived mode on top of it.
        r1 = float(np.clip(r[1], -0.999999, 0.999999))
        tau_ar1 = max((1.0 + r1) / (1.0 - r1), 1.0)
        ar1_excess = tau / tau_ar1

        ok_ess = ess >= ESS_MIN
        ok_rhat = not np.isnan(rhat) and rhat <= RHAT_MAX
        ok_drift = abs(drift) <= DRIFT_MAX
        ok_tail = ar1_excess <= AR1_EXCESS_MAX and converged

        if ok_ess and ok_rhat and ok_drift and ok_tail:
            verdict = "OK (slow but stationary)"
        elif not ok_tail:
            verdict = "LONG-LIVED MODE"
        elif not ok_drift or not ok_rhat:
            verdict = "NOT STATIONARY"
        else:
            verdict = "ESS TOO LOW"

        print(
            f"  [{lo:4d},{hi:4d}) {r[1]:7.3f} {tau:8.1f} {ess:7.1f} "
            f"{rhat:6.3f} {drift:7.2f} {ar1_excess:7.1f}  {verdict}"
        )
        rows.append((int(lo), int(hi), r[1], tau, ess, rhat, drift, ar1_excess, verdict))

    bad = [r for r in rows if not r[8].startswith("OK")]
    if bad:
        print(f"\n  => {len(bad)}/{len(rows)} bins fail: " + ", ".join(
            f"[{r[0]},{r[1]}) {r[8]}" for r in bad
        ))
        print(f"  => worst-case usable ESS across bins: {min(r[4] for r in rows):.1f}")
    else:
        print(
            f"\n  => ALL BINS USABLE. min ESS={min(r[4] for r in rows):.1f}, "
            f"thin by ~{int(max(r[3] for r in rows))} sweeps."
        )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    args = p.parse_args()
    for f in args.files:
        if os.path.exists(f):
            score_file(f)
        else:
            print(f"MISSING: {f}")


if __name__ == "__main__":
    main()
