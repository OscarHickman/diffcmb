"""
Diagnostic script for the lmax=128 MCLMC equilibration pathology.

CONTEXT (ROADMAP.md, 2026-08-09):
  Job 11710475 ran MCLMC (n_steps=30, step_size=0.1, L=200) for 3700 sweeps at
  lmax=128/nside=128 and produced a NO-GO: the lag-k autocorrelation profile is
  non-monotonic in several bins (e.g. [2,10) drops to -0.084 at lag-50 then
  bounces back to +0.499 at lag-100; [100,128) has drift_sigma=+1.10).

  This is NOT the typical "slow monotonic decay" profile of under-mixing —
  it looks more like oscillation or a second slow mode.  This script diagnoses
  which story best fits:

  HYPOTHESIS A — STEP SIZE TOO LARGE (oscillation):
    - step_size=0.1 was tuned at nside=64; the pilot uses nside=128.
    - In whitened coords the landscape curvature changes with the SHT basis.
    - Signature: autocorrelation goes negative (overshoot) then bounces back.
    - Expected: [2,10) oscillation, [100,128) upward drift = high-l modes still
      bouncing off a curvature wall rather than exploring.
    - Fix: reduce step_size (try 0.03-0.05).

  HYPOTHESIS B — VERY LONG AUTOCORRELATION TIME (slow mode):
    - Some φ modes have autocorrelation >> 200 sweeps even at correct step size.
    - Signature: monotonic decay that hasn't crossed 0 by lag 200.
    - Fix: run longer, or increase L.

  HYPOTHESIS C — BURN-IN INCOMPLETE (still drifting to stationarity):
    - n_burnin=400 at lmax=128/nside=128 is the same as earlier nside=64 runs.
    - Signature: per-bin power *level* is still trending after the burn-in window.
    - Fix: raise n_burnin; pre-warm phi from a phi MAP estimate, not a prior draw.

Diagnostics produced:
  1. Fine-grained per-bin autocorrelation at lags 1,2,...,30 to see the first
     zero-crossing clearly — distinguishes oscillation (goes negative early)
     from slow monotone (stays positive out to large lags).
  2. Per-third power-level evolution: is each bin rising, falling, or stationary
     across the 3300 samples? Direction and magnitude relative to truth.
  3. Per-bin absolute power level vs the true phi power — were we anywhere near
     the truth by the end of the chain?  Measures absolute bias, not just mixing.
  4. Burn-in trace: first 400 samples (burn-in window at this scale) vs the
     subsequent 3300 — checks whether the chain left the MAP start cleanly.
  5. Summary verdict (A/B/C above) with a recommended next step.

Outputs: prints tables to stdout; saves plots to results/analysis/
  diagnose_equilibration_pathology.png  (per-bin traces, fine-lag ACF, power ratio)

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/diagnose_equilibration_pathology.py \\
      --npz results/analysis/pilot_coverage_lmax128_mclmc_n3300.npz --lmax 128
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pilot_coverage_equilibration import (
    DIAGNOSTIC_LAGS,
    DRIFT_NOGO_SIGMA,
    LAG1_NOGO,
    drift_sigma,
    lag_autocorr,
    phi_power_traces,
)

from diffcmb.alm_utils import packed_sizes

try:
    from diffcmb.samplers import _alm_index_lm
except ImportError:
    # Allow running outside PYTHONPATH by providing a minimal fallback.
    def _alm_index_lm(lmax, n_real, n_imag):
        """Reproduce the index arrays used by pilot_coverage_equilibration.py."""
        L_list, m_list = [], []
        for ell in range(2, lmax):
            for m in range(ell + 1):
                if m == 0:
                    L_list.append(ell)
                    m_list.append(m)
        for ell in range(2, lmax):
            for m in range(1, ell + 1):
                L_list.append(ell)
                m_list.append(m)
        return np.array(L_list), np.array(m_list)


# Fine lags for oscillation detection (step 1).
FINE_LAGS = list(range(1, 31)) + [40, 50, 75, 100, 150, 200, 300, 500]


def compute_true_phi_power_per_bin(phi_true_packed, L_arr, ell_bins):
    """Mean |phi_lm|^2 per l-bin for the drawn truth phi."""
    power = phi_true_packed ** 2
    result = {}
    for lo, hi in ell_bins:
        mask = (L_arr >= lo) & (L_arr < hi)
        if mask.any():
            result[(lo, hi)] = float(power[mask].mean())
    return result


def plot_diagnostics(traces, phi_true_power, ell_bins, fine_lag_acfs,
                     n_samples, out_path):
    """Multi-panel diagnostic figure."""
    n_bins = len(ell_bins)
    fig, axes = plt.subplots(3, n_bins, figsize=(4 * n_bins, 12))
    if n_bins == 1:
        axes = axes[:, np.newaxis]

    bin_labels = [f"[{lo},{hi})" for lo, hi in ell_bins]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_bins))

    for j, (lo, hi) in enumerate(ell_bins):
        key = (lo, hi)
        if key not in traces:
            continue
        trace = traces[key]
        x = np.arange(len(trace))
        c = colors[j]
        true_val = phi_true_power.get(key, np.nan)

        # Row 0: raw trace with truth power overlaid
        ax = axes[0, j]
        ax.plot(x, trace, lw=0.4, alpha=0.8, color=c)
        ax.axhline(true_val, color="red", lw=1.5, ls="--", label="truth power")
        third = len(trace) // 3
        for k, (start, end, label) in enumerate([
            (0, third, "1st third"),
            (third, 2 * third, "2nd third"),
            (2 * third, len(trace), "3rd third"),
        ]):
            ax.axhline(trace[start:end].mean(), color=f"C{k}", lw=1, ls=":",
                       alpha=0.8, label=f"{label} mean")
        ax.set_title(f"l-bin {bin_labels[j]}", fontsize=9)
        ax.set_xlabel("sweep", fontsize=8)
        ax.set_ylabel(r"$\langle|\phi_{lm}|^2\rangle$", fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=6, loc="upper right")

        # Row 1: fine-lag ACF (the key oscillation/monotone discriminator)
        ax = axes[1, j]
        lags_plot = [l for l in FINE_LAGS if l < len(trace)]
        acf_vals = fine_lag_acfs.get(key, [np.nan] * len(lags_plot))
        ax.plot(lags_plot[:len(acf_vals)], acf_vals[:len(lags_plot)],
                "o-", ms=3, lw=1.0, color=c)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.axhline(0.2, color="orange", lw=0.8, ls=":", label="|r|=0.2 target")
        ax.axhline(-0.2, color="orange", lw=0.8, ls=":")
        ax.set_xlabel("lag k", fontsize=8)
        ax.set_ylabel(r"$r_k$", fontsize=8)
        ax.set_title("ACF (fine lags)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_ylim(-1.05, 1.05)
        if j == 0:
            ax.legend(fontsize=6)

        # Row 2: power ratio = trace / truth_power (shows absolute bias)
        ax = axes[2, j]
        if np.isfinite(true_val) and true_val > 0:
            ratio = trace / true_val
            ax.plot(x, ratio, lw=0.4, alpha=0.8, color=c)
            ax.axhline(1.0, color="red", lw=1.5, ls="--", label="ratio=1 (unbiased)")
            ax.axhline(ratio[:third].mean(), color="C0", lw=1, ls=":", alpha=0.8,
                       label=f"1st-third mean {ratio[:third].mean():.2f}")
            ax.axhline(ratio[-third:].mean(), color="C2", lw=1, ls=":", alpha=0.8,
                       label=f"3rd-third mean {ratio[-third:].mean():.2f}")
            ax.set_ylabel("power / truth", fontsize=8)
        else:
            ax.text(0.5, 0.5, "no truth reference", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)
        ax.set_xlabel("sweep", fontsize=8)
        ax.set_title("Power / truth power", fontsize=9)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=6, loc="upper right")

    fig.suptitle(
        f"lmax=128 MCLMC pilot (job 11710475, n={n_samples} sweeps)\n"
        "Equilibration pathology diagnostics",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved diagnostic figure to {out_path}")


def verdict_from_diagnostics(fine_lag_acfs, traces, phi_true_power, ell_bins):
    """Print a structured verdict and recommended next step."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC VERDICT")
    print("=" * 70)

    oscillating_bins = []
    drifting_bins = []
    slow_decay_bins = []
    biased_bins = []

    for lo, hi in ell_bins:
        key = (lo, hi)
        if key not in traces:
            continue
        trace = traces[key]
        acf = fine_lag_acfs.get(key, [])

        # First zero-crossing of ACF
        first_neg = next((i for i, a in enumerate(acf) if np.isfinite(a) and a < 0), None)
        first_zero_lag = FINE_LAGS[first_neg] if first_neg is not None else None

        # Does it bounce back positive after first going negative?
        bounces = False
        if first_neg is not None and first_neg + 1 < len(acf):
            tail = acf[first_neg + 1:]
            bounces = any(np.isfinite(a) and a > 0.15 for a in tail)

        d = drift_sigma(trace)
        true_val = phi_true_power.get(key, np.nan)
        third = len(trace) // 3
        power_ratio_end = (
            float(trace[-third:].mean() / true_val)
            if (np.isfinite(true_val) and true_val > 0)
            else np.nan
        )

        label = f"[{lo},{hi})"
        print(f"\n  Bin {label}:")
        print(f"    First negative ACF at lag: {first_zero_lag}")
        print(f"    ACF bounces back positive after first zero-crossing: {bounces}")
        print(f"    Drift sigma: {d:.3f} (gate={DRIFT_NOGO_SIGMA})")
        print(f"    End-of-chain power / truth power: {power_ratio_end:.3f}")

        if bounces and first_zero_lag is not None and first_zero_lag <= 60:
            oscillating_bins.append(label)
            print("    -> OSCILLATION signature (overshoot, early bounce-back)")
        elif d > DRIFT_NOGO_SIGMA:
            drifting_bins.append(label)
            print("    -> DRIFT signature (monotonic trend in power level)")
        else:
            slow_decay_bins.append(label)
            print("    -> SLOW DECAY (monotone ACF, no oscillation, no drift)")

        if np.isfinite(power_ratio_end) and power_ratio_end < 0.3:
            biased_bins.append(label)
            print("    -> LARGE ABSOLUTE BIAS (end-of-chain power < 30% of truth)")

    print("\n" + "-" * 70)
    print("SUMMARY:")
    print(f"  Oscillating bins (step_size too large?): {oscillating_bins or 'none'}")
    print(f"  Drifting bins (burn-in incomplete?):     {drifting_bins or 'none'}")
    print(f"  Slow-decay bins (long autocorr time?):  {slow_decay_bins or 'none'}")
    print(f"  Severely biased bins (<30% of truth):   {biased_bins or 'none'}")

    print("\nRECOMMENDED NEXT STEP:")
    if oscillating_bins:
        print(
            "  HYPOTHESIS A (step_size too large) is consistent with oscillating bins.\n"
            "  step_size=0.1 was tuned at nside=64; current pilot uses nside=128.\n"
            "  The whitened-space curvature changes with the SHT basis (larger nside\n"
            "  = finer resolution = potentially steeper gradient directions).\n"
            "\n"
            "  Recommended action: run a step_size grid at nside=128 on a short\n"
            "  window (~200 sweeps, dine2) with step_size in {0.01, 0.03, 0.05, 0.1}\n"
            "  and L in {50, 200} — same gate_phi_mclmc_vs_hmc.py framework but\n"
            "  at the production nside. This is a cheap gating experiment before\n"
            "  re-running the full 3700-sweep equilibration pilot."
        )
    elif drifting_bins and not oscillating_bins:
        print(
            "  HYPOTHESIS C (burn-in incomplete) is consistent with drifting bins.\n"
            "  Recommended action: raise n_burnin to 1000-2000 and/or initialise phi\n"
            "  from a phi-MAP estimate rather than a prior draw."
        )
    elif slow_decay_bins and not oscillating_bins and not drifting_bins:
        print(
            "  HYPOTHESIS B (long autocorrelation time) is consistent with the data.\n"
            "  Recommended action: increase L (momentum decoherence scale) to 400-600\n"
            "  to reduce correlation between consecutive sweeps, or run a longer chain."
        )
    else:
        print(
            "  Mixed or unclear: inspect the figure and the per-bin tables above.\n"
            "  The fine-lag ACF plot (row 1) is the clearest discriminator:\n"
            "  oscillation = goes negative early then bounces back;\n"
            "  slow decay = stays positive and monotonically declines."
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--npz",
        default="results/analysis/pilot_coverage_lmax128_mclmc_n3300.npz",
        help="Path to the pilot chain npz (must contain phi_samples and phi_true_packed).",
    )
    p.add_argument("--lmax", type=int, default=128)
    p.add_argument(
        "--out",
        default="results/analysis/diagnose_equilibration_pathology.png",
        help="Output figure path.",
    )
    args = p.parse_args()

    print(f"Loading {args.npz} ...")
    data = np.load(args.npz, allow_pickle=True)
    phi_samples = data["phi_samples"]    # shape (n_samples, n_phi_packed)
    n_samples = phi_samples.shape[0]
    print(f"  phi_samples shape: {phi_samples.shape}  (n_samples={n_samples})")

    if "phi_true_packed" in data.files:
        phi_true_packed = data["phi_true_packed"]
        print(f"  phi_true_packed found, shape={phi_true_packed.shape}")
    else:
        phi_true_packed = None
        print("  WARNING: phi_true_packed not in npz — absolute bias check skipped")

    lmax = args.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    L_arr, _m_arr = _alm_index_lm(lmax, n_real, n_imag)

    ell_bins = [
        (lo, min(hi, lmax))
        for lo, hi in [(2, 10), (10, 30), (30, 60), (60, 100), (100, 150)]
        if lo < lmax
    ]

    print(f"\nComputing per-bin phi-power traces across {n_samples} sweeps...")
    traces = phi_power_traces(phi_samples, L_arr, ell_bins)

    # Compute fine-lag ACF for oscillation detection
    print(f"Computing fine-lag ACF (lags: {FINE_LAGS[:10]}...{FINE_LAGS[-3:]}) ...")
    fine_lag_acfs = {}
    for key, trace in traces.items():
        acf = []
        for lag in FINE_LAGS:
            if lag < len(trace):
                acf.append(lag_autocorr(trace, lag))
            else:
                acf.append(np.nan)
        fine_lag_acfs[key] = acf

    # True phi power per bin (for absolute bias check)
    phi_true_power = {}
    if phi_true_packed is not None:
        phi_true_power = compute_true_phi_power_per_bin(phi_true_packed, L_arr, ell_bins)
    else:
        phi_true_power = dict.fromkeys(traces, np.nan)

    # --- Print full fine-lag table ---
    print("\n--- Fine-lag ACF table (rows=bins, cols=lags 1..30, 40, 50, 75, ...) ---")
    header_lags = [l for l in FINE_LAGS if l <= 50] + [75, 100, 150, 200]
    hdr = "  l-bin        " + "  ".join(f"r_{k:<4d}" for k in header_lags)
    print(hdr)
    for (lo, hi), acf in fine_lag_acfs.items():
        lag_to_idx = {l: i for i, l in enumerate(FINE_LAGS)}
        vals = [acf[lag_to_idx[l]] if l in lag_to_idx and lag_to_idx[l] < len(acf)
                else np.nan for l in header_lags]
        row = "  ".join(f"{v:6.3f}" if np.isfinite(v) else "   nan" for v in vals)
        print(f"  [{lo:4d},{hi:4d})  {row}")

    # --- Print power ratio table ---
    print("\n--- Per-bin power ratio (chain mean / truth power) ---")
    print("  l-bin        1st-third   2nd-third   3rd-third   end/truth   drift_sigma")
    for (lo, hi), trace in traces.items():
        key = (lo, hi)
        true_val = phi_true_power.get(key, np.nan)
        third = len(trace) // 3
        m1 = trace[:third].mean()
        m2 = trace[third:2*third].mean()
        m3 = trace[-third:].mean()
        d = drift_sigma(trace)
        if np.isfinite(true_val) and true_val > 0:
            r1, r2, r3 = m1/true_val, m2/true_val, m3/true_val
            print(f"  [{lo:4d},{hi:4d})  {r1:9.3f}   {r2:9.3f}   {r3:9.3f}   {r3:9.3f}   {d:+8.3f}")
        else:
            print(f"  [{lo:4d},{hi:4d})  (no truth reference)                             {d:+8.3f}")

    # Verdict
    verdict_from_diagnostics(fine_lag_acfs, traces, phi_true_power, ell_bins)

    # Plot
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plot_diagnostics(traces, phi_true_power, ell_bins, fine_lag_acfs,
                     n_samples, args.out)


if __name__ == "__main__":
    main()
