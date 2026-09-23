"""ROADMAP T0.1 (a, b): is the exact-operator calibration failure thinning,
burn-in, or the sampler?

The 2026-09-23 harvest rejected the field-power ranks (a_lm `[30,60)` in BOTH
ensembles, phi in Block-4-ON) at thin 10, where the a_lm bulk ESS is only ~35
per 1200 sweeps. Two readings, which this script separates:

  (a) THINNING. Autocorrelated draws inflate the SPREAD of the ranks (U-shape)
      but leave the rank MEAN unbiased -- if the chain is stationary. So the
      field ranks are re-scored at thin 10 and thin 40 (~tau_int), and the
      mean and spread p-values are reported separately. A spread failure that
      goes away at thin 40 was autocorrelation; a MEAN offset that survives
      thinning is not a thinning artefact.
  (b) STATIONARITY. The chains start from a MAP estimate after 400 burn-in
      sweeps. If burn-in is too short, the posterior power drifts toward its
      stationary value along the saved chain, and the rank mean of the truth
      computed from each QUARTER of the chain moves with sweep number. A
      mean_u that is flat across quarters rules burn-in out and indicts the
      sampler or the statistic; one that trends toward 0.5 says run longer.

Everything here is numpy on the saved chains; the joint-likelihood SBC (which
needs the model) is run separately by `sbc_joint_likelihood.py --thin 40`
and `--burn_frac 0.5` from the same SLURM wrapper.

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/diagnose_calibration_stationarity.py \
      --indir results/analysis/ens_exact_l64_A3000_n30 --thins 10,40 --n_seg 4
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper"))

from aggregate_coverage_ranks import binned_power, rank_of  # noqa: E402
from fig1_validation import ELL_BINS, mean_sd_p  # noqa: E402
from fig_convergence import bulk_ess  # noqa: E402

from diffcmb.alm_utils import packed_length, packed_sizes  # noqa: E402
from diffcmb.samplers import _alm_index_lm  # noqa: E402

WINDOWS = ("full", "first", "second")


def chain_files(indir):
    files = sorted(glob.glob(os.path.join(indir, "chain_r[0-9][0-9][0-9].npz")))
    if not files:
        raise SystemExit(f"no chains in {indir}")
    return files


def window_slice(n, window):
    """Sweep range of the saved chain for 'full', 'first' or 'second' half."""
    half = n // 2
    return {"full": slice(0, n), "first": slice(0, half),
            "second": slice(half, n)}[window]


def power_traces(d):
    """Per-sweep binned field power / truth power, for every (tag, bin).

    Returns {(tag, lo, hi): trace} where trace has one entry per saved sweep.
    Dividing by the truth makes chains comparable (a correct sampler's trace
    scatters around ~1 with realization-dependent offset); the rank of the
    truth is rank of 1.0 in the trace.
    """
    lmax = int(d["lmax"])
    want = packed_length(lmax)
    alm_true = d["alm_true_packed"]
    if alm_true.shape[0] != want:
        raise SystemExit(f"field vector {alm_true.shape[0]} != packed_length({lmax})"
                         f"={want}: pre-2026-09-06 packing, a different model")
    n_real, n_imag = packed_sizes(lmax)
    L_arr, _m = _alm_index_lm(lmax, n_real, n_imag)
    fields = (("phi_power", d["phi_samples"], d["phi_true_packed"]),
              ("alm_power", d["alm_samples"][:, lmax - 2:], alm_true))
    out = {}
    for lo, hi in ELL_BINS:
        hi = min(hi, lmax)
        if lo >= hi:
            continue
        for tag, samp, truth in fields:
            post = binned_power(samp, L_arr, lo, hi)
            if post is not None:
                out[(tag, lo, hi)] = post / binned_power(truth, L_arr, lo, hi)
    return out


def load_traces(files):
    """[{(tag,lo,hi): trace, 'logp': logp}] for every chain."""
    traces = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        t = power_traces(d)
        t["logp"] = np.asarray(d["logp"], dtype=np.float64)
        traces.append(t)
    return traces


def rank_table(traces, thin, window):
    """{key: (ranks, n_draws)} -- rank of the truth (ratio 1.0) among the
    thinned draws of the chosen half."""
    table = {}
    for t in traces:
        for key, tr in t.items():
            if key == "logp":
                continue
            s = tr[window_slice(len(tr), window)][::thin]
            table.setdefault(key, [[], len(s) - 1])[0].append(rank_of(1.0, s))
    return {k: (np.asarray(r), n) for k, (r, n) in table.items()}


def segment_mean_u(traces, n_seg):
    """{key: (n_chain, n_seg) array of u = rank/n within each chain segment}.

    Uses every sweep of a segment: the rank MEAN is unbiased under
    autocorrelation for a stationary chain, so no thinning is needed for it.
    """
    out = {}
    for t in traces:
        for key, tr in t.items():
            if key == "logp":
                continue
            segs = np.array_split(tr, n_seg)
            out.setdefault(key, []).append(
                [(np.sum(s < 1.0) + 0.5) / (len(s) + 1.0) for s in segs])
    return {k: np.asarray(v) for k, v in out.items()}


def segment_level(traces, key, n_seg):
    """(n_chain, n_seg) mean of log(trace) per segment, and of logp."""
    rows = []
    for t in traces:
        tr = t[key] if key == "logp" else np.log(t[key])
        rows.append([s.mean() for s in np.array_split(tr, n_seg)])
    return np.asarray(rows)


def drift_z(levels):
    """Across-chain z of (last segment - first segment); ~N(0,1) if stationary.

    Chains are independent realizations, so the across-chain standard error of
    the per-chain difference is an honest error bar regardless of
    within-chain autocorrelation.
    """
    diff = levels[:, -1] - levels[:, 0]
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    return float(diff.mean() / se) if se > 0 else 0.0


def tau_int(traces, key):
    """Median over chains of n / bulk ESS for one scalar trace."""
    taus = [len(t[key]) / float(bulk_ess(np.log(t[key])[:, None])[0])
            for t in traces]
    return float(np.median(taus))


def report(indir, thins, n_seg):
    files = chain_files(indir)
    traces = load_traces(files)
    keys = sorted(k for k in traces[0] if k != "logp")
    n = len(traces[0]["logp"])
    print(f"\n######## {indir}: {len(files)} chains x {n} sweeps ########")

    print("\n--- (a) field ranks by thinning and half: mean_u (p_mean) | sd_u (p_sd) ---")
    print("    uniform: mean_u 0.5, sd_u 0.289. A MEAN offset that survives thinning")
    print("    is not autocorrelation; a spread excess that vanishes at thin~tau was.")
    for thin in thins:
        for window in WINDOWS:
            table = rank_table(traces, thin, window)
            print(f"\n  thin={thin} window={window}")
            for key in keys:
                ranks, nd = table[key]
                u = (ranks + 0.5) / (nd + 1.0)
                p_m, p_s = mean_sd_p(ranks, nd, n_rep=5000)
                print(f"    {key[0]:9s} [{key[1]:2d},{key[2]:2d})  draws {nd + 1:4d}  "
                      f"mean_u {u.mean():.3f} (p {p_m:.3f}) | sd_u {u.std():.3f} "
                      f"(p {p_s:.3f})")

    print(f"\n--- (b) stationarity: rank mean_u of the truth per chain {n_seg}-segment ---")
    print("    flat across segments => not burn-in; trending toward 0.5 => run longer.")
    seg_u = segment_mean_u(traces, n_seg)
    se_u = 0.289 / np.sqrt(len(files))
    for key in keys:
        cols = "  ".join(f"{m:.3f}" for m in seg_u[key].mean(axis=0))
        print(f"    {key[0]:9s} [{key[1]:2d},{key[2]:2d})  {cols}   (+/- {se_u:.3f})")

    print("\n--- (b) drift of log(power/truth) and logp, last vs first segment ---")
    print("    z ~ N(0,1) across independent chains if stationary; tau_int = n/ESS.")
    for key in keys + ["logp"]:
        lev = segment_level(traces, key, n_seg)
        cols = "  ".join(f"{m:+.4f}" for m in (lev.mean(axis=0) - lev[:, 0].mean()))
        name = "logp" if key == "logp" else f"{key[0]:9s} [{key[1]:2d},{key[2]:2d})"
        tau = "" if key == "logp" else f"  tau_int {tau_int(traces, key):6.1f}"
        print(f"    {name:20s} rel. to seg 0: {cols}   drift z {drift_z(lev):+.2f}{tau}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indir", action="append", required=True,
                    help="ensemble directory; repeat for several")
    ap.add_argument("--thins", default="10,40")
    ap.add_argument("--n_seg", type=int, default=4)
    args = ap.parse_args()
    thins = [int(t) for t in args.thins.split(",")]
    for indir in args.indir:
        report(indir, thins, args.n_seg)


if __name__ == "__main__":
    main()
