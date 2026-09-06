"""
Offline re-analysis of the lmax=128 coverage-pilot checkpoint after its
window-extension job (11665429) OOM'd at 1800/2500 samples (up from the 600
that originally triggered NO-GO in job 11663477 -- see ROADMAP.md/achievements.md).

Reuses scripts/pilot_coverage_equilibration.py's own lag-k/drift diagnostics
and GO/NO-GO verdict logic verbatim -- no new statistics -- against the
checkpoint's phi_samples directly, so no chain needs to be re-run.

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/reanalyze_pilot_checkpoint.py \
    --checkpoint results/analysis/pilot_coverage_lmax128_v2_ckpt.npz --lmax 128
"""
import argparse

import numpy as np
from pilot_coverage_equilibration import (
    DRIFT_NOGO_SIGMA,
    LAG1_NOGO,
    PHI_ACCEPT_NOGO,
    phi_power_traces,
    report_equilibration,
)

from diffcmb.alm_utils import packed_sizes
from diffcmb.samplers import _alm_index_lm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str,
                   default="results/analysis/pilot_coverage_lmax128_v2_ckpt.npz")
    p.add_argument("--lmax", type=int, default=128)
    args = p.parse_args()
    lmax = args.lmax

    ck = np.load(args.checkpoint, allow_pickle=True)
    phi_samples = ck["phi_samples"]
    phi_accept = float(np.mean(ck["phi_accepts"])) if "phi_accepts" in ck.files else np.nan
    n_collected = phi_samples.shape[0]
    print(f"Loaded {n_collected} samples from {args.checkpoint} (lmax={lmax})")
    print(f"  alm-block accept rate: {float(np.mean(ck['accepts'])):.3f}")
    print(f"  phi block accept rate: {phi_accept:.4f} (gate floor is {PHI_ACCEPT_NOGO})")

    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    L_arr, _m_arr = _alm_index_lm(lmax, n_real, n_imag)

    ell_bins = [(lo, min(hi, lmax)) for lo, hi in
                [(2, 10), (10, 30), (30, 60), (60, 100), (100, 150)]
                if lo < lmax]

    traces = phi_power_traces(phi_samples, L_arr, ell_bins)
    phi_rows = report_equilibration(traces, f"phi block (lensing potential), n={n_collected}")

    lag1 = np.array([r[2] for r in phi_rows], dtype=np.float64)
    drifts = np.array([r[-1] for r in phi_rows], dtype=np.float64)
    worst_lag1 = np.nanmax(lag1) if len(lag1) else np.nan
    worst_drift = np.nanmax(np.abs(drifts)) if len(drifts) else np.nan
    n_nan = int(np.sum(~np.isfinite(lag1)) + np.sum(~np.isfinite(drifts)))

    print("\n=== Verdict (re-analysis of extended checkpoint) ===")
    if np.isfinite(phi_accept) and phi_accept < PHI_ACCEPT_NOGO:
        print(f"NO-GO: phi accept rate {phi_accept:.4f} < {PHI_ACCEPT_NOGO}.")
    elif n_nan:
        print(f"NO-GO / INCONCLUSIVE: {n_nan} non-finite diagnostic(s).")
    elif worst_lag1 >= LAG1_NOGO or worst_drift >= DRIFT_NOGO_SIGMA:
        reasons = []
        if worst_lag1 >= LAG1_NOGO:
            reasons.append(f"worst lag-1 autocorrelation {worst_lag1:.3f} >= {LAG1_NOGO}")
        if worst_drift >= DRIFT_NOGO_SIGMA:
            reasons.append(f"worst |drift| {worst_drift:.2f} sigma >= {DRIFT_NOGO_SIGMA}")
        print(f"NO-GO (fixed lag-1 gate): {'; '.join(reasons)}. See the lag-k table "
              f"above for whether this is genuine decay-to-zero (judgment-call GO "
              f"territory per ROADMAP.md) or a stuck chain.")
    else:
        print(f"GO: worst lag-1 autocorrelation {worst_lag1:.3f} (< {LAG1_NOGO}) and "
              f"worst |drift| {worst_drift:.2f} sigma (< {DRIFT_NOGO_SIGMA}) at "
              f"n={n_collected} samples.")


if __name__ == "__main__":
    main()
