"""Free re-analysis (no new sampling): test whether phi-block equilibration
failure tracks distance from the harmonic band-limit (lmax) rather than
absolute ell.

Reuses phi_traces/phi_trace_bins already saved by every pilot run (no new
compute) and recomputes lag-1 autocorrelation + drift with the exact same
functions as scripts/archive/pilot_coverage_equilibration.py, across all four
phi-equilibration pilots run so far at two different lmax values. If the
failing bins collapse onto a common (lmax - bin_hi) trend, that's evidence
for band-edge conditioning rather than a per-integrator/per-geometry defect.
"""
import numpy as np

RUNS = [
    ("lmax128_hmc_phi240 (job 11694912)",
     "results/analysis/pilot_coverage_lmax128_v3_phi240_n3300.npz"),
    ("lmax128_mclmc_untuned (job 11710475)",
     "results/analysis/pilot_coverage_lmax128_mclmc_n3300.npz"),
    ("lmax128_mclmc_tuned_v2 (job 11744132)",
     "results/analysis/pilot_coverage_lmax128_mclmc_v2_n3300.npz"),
    ("lmax128_mclmc_fisher (job 11752451)",
     "results/analysis/diagnose_mclmc_fisher_geometry_n3300.npz"),
    ("lmax64_hmc_phi240 (job 11752452)",
     "results/analysis/pilot_coverage_lmax64_hmc_n3300.npz"),
]

DIAGNOSTIC_LAGS = [1, 5, 10, 25, 50, 75, 100, 150, 200]


def lag_autocorr(series: np.ndarray, lag: int) -> float:
    n = len(series)
    if n <= lag + 1:
        return float("nan")
    x = np.asarray(series, dtype=np.float64)
    x0 = x[:-lag] - x.mean()
    x1 = x[lag:] - x.mean()
    denom = np.sum((x - x.mean()) ** 2)
    if denom <= 0:
        return float("nan")
    return float(np.sum(x0 * x1) / denom)


def drift_sigma(series: np.ndarray) -> float:
    n = len(series)
    if n < 9:
        return float("nan")
    third = n // 3
    first, last = series[:third], series[-third:]
    scatter = np.sqrt(0.5 * (first.var(ddof=1) + last.var(ddof=1)))
    if scatter <= 0:
        return float("nan")
    return float((last.mean() - first.mean()) / scatter)


def main() -> None:
    rows = []
    for label, path in RUNS:
        d = np.load(path, allow_pickle=True)
        lmax = int(d["lmax"])
        bins = d["phi_trace_bins"]
        traces = d["phi_traces"]
        for (lo, hi), trace in zip(bins, traces):
            r1 = lag_autocorr(trace, 1)
            r200 = lag_autocorr(trace, 200)
            drift = drift_sigma(trace)
            dist = lmax - hi
            rows.append((label, lmax, lo, hi, dist, r1, r200, drift))

    print(f"{'run':<42} {'lmax':>5} {'bin':>10} {'lmax-hi':>8} "
          f"{'r_1':>7} {'r_200':>7} {'drift':>8}")
    for label, lmax, lo, hi, dist, r1, r200, drift in rows:
        print(f"{label:<42} {lmax:>5} [{lo:4d},{hi:4d}) {dist:>8} "
              f"{r1:>7.3f} {r200:>7.3f} {drift:>8.2f}")

    print("\n--- grouped by distance-from-cutoff (lmax - bin_hi) ---")
    by_dist: dict[int, list[tuple]] = {}
    for label, _lmax, _lo, _hi, dist, r1, r200, drift in rows:
        by_dist.setdefault(dist, []).append((label, r1, r200, drift))
    for dist in sorted(by_dist):
        entries = by_dist[dist]
        mean_r1 = np.nanmean([e[1] for e in entries])
        mean_r200 = np.nanmean([e[2] for e in entries])
        mean_drift = np.nanmean([abs(e[3]) for e in entries])
        print(f"  lmax-hi={dist:>4}  n={len(entries):>2}  "
              f"mean r_1={mean_r1:.3f}  mean r_200={mean_r200:.3f}  "
              f"mean |drift|={mean_drift:.2f}")


if __name__ == "__main__":
    main()
