"""
MCLMC step_size / L tuning grid at the *production* nside=128.

CONTEXT (ROADMAP.md, achievements.md):
  Job 11710475 (MCLMC, n_steps=30, step_size=0.1, L=200, nside=128, 3700 sweeps)
  returned NO-GO on the lag-1<0.2-by-lag-200 equilibration gate.
  The autocorrelation profile in bin [2,10) is non-monotonic (goes negative at
  lag~50, bounces back positive at lag~100), strongly suggesting Hypothesis A:
  step_size=0.1 was tuned at nside=64 (job 11708844); the whitened-space curvature
  at nside=128 is different, and step_size=0.1 may be causing oscillation.

  This script runs a short (~200-sweep) step_size x L grid at the production
  nside=128 to find a step_size that gives monotone ACF decay rather than oscillation.
  It is deliberately scoped as a *bounded tuning experiment*, not a full equilibration
  test: we want the fine-lag ACF profile (does it go negative early?) rather than
  a converged posterior.

DESIGN NOTES:
  - Uses use_matrixfree_sht=True (same as the production pilot) so the whitened
    coordinates match exactly — this is what the step_size=0.1 mismatch is about.
  - Starts phi from a prior draw (not truth-warm, since we only care about the
    ACF shape in the stationary regime, and truth-warm-start would contaminate
    the ACF measurement).
  - Runs a short n_burnin=150, n_samples=200 per grid point. Enough to see the
    first ~100 lag autocorrelations; not enough to fully equilibrate.
  - For each (step_size, L) pair: reports the per-bin lag-1, fine-lag first-zero-
    crossing, and whether the ACF goes negative (oscillation) or stays positive
    (slow decay). The target configuration is the one with the smallest step_size
    that does NOT go negative in the [2,10) bin.
  - Saves a full npz per configuration for offline inspection.
  - Does NOT run the HMC baseline (already benchmarked in job 11708844).

Grid to search (informed by the oscillation hypothesis):
  step_size: 0.005, 0.01, 0.02, 0.05  (going well below the current 0.1)
  L:         100, 200                   (L=200 was the production value; try L=100
                                         as a tighter decoherence to compensate
                                         for the shorter step)
  n_steps:   30  (keep fixed; optimal from job 11708844)

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/tune_mclmc_step_size_nside128.py \\
      --lmax 128 --nside 128 \\
      --step_sizes 0.005,0.01,0.02,0.05 \\
      --L_vals 100,200 \\
      --n_steps 30 --n_burnin 150 --n_samples 200 \\
      --out_dir results/analysis/mclmc_tune_nside128
"""

import argparse
import os
import sys
import time

import healpy as hp
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diffcmb import CosmologyAdvancedSampling, run_gibbs_chain
from diffcmb.alm_utils import packed_sizes
from diffcmb.lensing import _alm_hp_to_packed, compute_sl_phi_np, lens_map_tf
from diffcmb.power import call_CAMB_map
from diffcmb.samplers import _alm_index_lm, find_map_estimate

LCDM_PARAMS = [67.74, 0.0486, 0.2589, 0.06, 0.0, 0.066]

FINE_LAGS = list(range(1, 31)) + [40, 50, 75, 100, 150, 200]


def get_cl_phiphi(lmax):
    import camb
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=LCDM_PARAMS[0], ombh2=LCDM_PARAMS[1], omch2=LCDM_PARAMS[2],
        mnu=LCDM_PARAMS[3], omk=LCDM_PARAMS[4], tau=LCDM_PARAMS[5],
    )
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    results = camb.get_results(pars)
    dl_pp = results.get_lens_potential_cls(lmax=lmax - 1)[:, 0]
    cl_pp = np.zeros(lmax, dtype=np.float64)
    for ell in range(2, lmax):
        if ell < len(dl_pp):
            norm = (ell * (ell + 1)) ** 2 / (2.0 * np.pi)
            cl_pp[ell] = dl_pp[ell] / norm
    return cl_pp


def lag_autocorr(series, lag):
    n = len(series)
    if n <= lag + 1:
        return np.nan
    x = np.asarray(series, dtype=np.float64)
    x0 = x[:-lag] - x.mean()
    x1 = x[lag:] - x.mean()
    denom = np.sum((x - x.mean()) ** 2)
    if denom <= 0:
        return np.nan
    return float(np.sum(x0 * x1) / denom)


def phi_power_traces(phi_samples, L_arr, ell_bins):
    power = phi_samples ** 2
    return {(lo, hi): power[:, (L_arr >= lo) & (L_arr < hi)].mean(axis=1)
            for lo, hi in ell_bins
            if ((L_arr >= lo) & (L_arr < hi)).any()}


def acf_profile(trace, fine_lags):
    return [lag_autocorr(trace, k) for k in fine_lags if k < len(trace)]


def report_config(step_size, L, n_steps, phi_samples, L_arr, ell_bins, wall_s, n_samples):
    """Print the fine-lag ACF table for one (step_size, L) config."""
    traces = phi_power_traces(phi_samples, L_arr, ell_bins)
    print(f"\n  step_size={step_size:.4g}  L={L:.4g}  n_steps={n_steps}  "
          f"wall={wall_s:.0f}s  ({wall_s/n_samples:.1f}s/sweep)")
    print(f"  {'l-bin':<12}  {'r1':>6}  {'first<0':>9}  {'bounces':>9}  {'first<0.2':>10}")
    summary = {}
    for (lo, hi), trace in sorted(traces.items()):
        acf = acf_profile(trace, FINE_LAGS)
        r1 = acf[0] if acf else np.nan
        first_neg_idx = next((i for i, a in enumerate(acf) if np.isfinite(a) and a < 0), None)
        first_neg_lag = FINE_LAGS[first_neg_idx] if first_neg_idx is not None else None
        # Bounces back positive after first zero crossing?
        bounces = False
        if first_neg_idx is not None and first_neg_idx + 1 < len(acf):
            bounces = any(np.isfinite(a) and a > 0.15 for a in acf[first_neg_idx + 1:])
        first_below02_idx = next((i for i, a in enumerate(acf) if np.isfinite(a) and abs(a) < 0.2), None)
        first_below02_lag = FINE_LAGS[first_below02_idx] if first_below02_idx is not None else None

        oscillates = (first_neg_lag is not None and first_neg_lag <= 60 and bounces)
        flag = " <-- OSCILLATES" if oscillates else (" <-- SLOW DECAY" if first_neg_lag is None else "")

        print(f"  [{lo:4d},{hi:4d})    "
              f"{r1:6.3f}  "
              f"{'lag '+str(first_neg_lag) if first_neg_lag else 'none':>9}  "
              f"{'yes' if bounces else 'no':>9}  "
              f"{'lag '+str(first_below02_lag) if first_below02_lag else '>200':>10}"
              f"{flag}")
        summary[(lo, hi)] = {
            "r1": r1, "first_neg_lag": first_neg_lag,
            "bounces": bounces, "oscillates": oscillates,
            "first_below02_lag": first_below02_lag,
        }
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lmax", type=int, default=128)
    p.add_argument("--nside", type=int, default=128)
    p.add_argument("--noisesig", type=float, default=1.0)
    p.add_argument("--n_burnin", type=int, default=150)
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--n_steps", type=int, default=30,
                   help="MCLMC n_steps per sweep (keep at 30, optimal from job 11708844)")
    p.add_argument("--step_sizes", type=str, default="0.005,0.01,0.02,0.05",
                   help="Comma-separated step_size values to try")
    p.add_argument("--L_vals", type=str, default="100,200",
                   help="Comma-separated L (momentum decoherence) values to try")
    p.add_argument("--map_steps", type=int, default=2000)
    p.add_argument("--map_lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="results/analysis/mclmc_tune_nside128")
    args = p.parse_args()

    lmax, nside = args.lmax, args.nside
    step_sizes = [float(s) for s in args.step_sizes.split(",")]
    L_vals = [float(s) for s in args.L_vals.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"=== MCLMC step_size/L tuning grid at nside={nside} (production scale) ===")
    print(f"  Grid: step_sizes={step_sizes}, L_vals={L_vals}, n_steps={args.n_steps}")
    print(f"  n_burnin={args.n_burnin}, n_samples={args.n_samples} per config\n")

    # Build model — matrix-free SHT, matching the production pilot exactly
    print("Building matrix-free-SHT model (synthetic, full-sky)...")
    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=args.noisesig,
        data_mode="synthetic", dtype=tf.complex128, use_matrixfree_sht=True,
    )
    model._ensure_tf_tensors()
    assert len(model.unmasked_idx) == model.NPIX, "tuning run assumes full-sky"

    print("Drawing (alm_true, phi_true) from CAMB spectra...")
    cl_true = call_CAMB_map(LCDM_PARAMS, lmax)
    cl_phiphi_true = get_cl_phiphi(lmax)

    rng_truth = np.random.default_rng(args.seed)
    rng_start = np.random.default_rng(args.seed + 10_000)
    rng_noise = np.random.default_rng(args.seed + 20_000)

    np.random.seed(rng_truth.integers(0, 2**31 - 1))
    alm_true_hp = hp.synalm(cl_true, lmax=lmax - 1, new=True).astype(np.complex128)
    phi_true_hp = hp.synalm(cl_phiphi_true, lmax=lmax - 1, new=True).astype(np.complex128)
    alm_true_packed = _alm_hp_to_packed(alm_true_hp, lmax)
    phi_true_packed = _alm_hp_to_packed(phi_true_hp, lmax)

    print("Lensing and adding noise...")
    T_lensed = lens_map_tf(
        model, tf.constant(alm_true_packed, tf.float64), phi_true_hp
    ).numpy()
    noisy_map = T_lensed + rng_noise.normal(0.0, args.noisesig, size=model.NPIX)
    model.prior_map = noisy_map
    model.prior_map_masked = tf.convert_to_tensor(
        noisy_map[model.unmasked_idx], dtype=tf.float64
    )

    # phi start: independent prior draw (not truth-warm)
    np.random.seed(rng_start.integers(0, 2**31 - 1))
    phi_start_hp = hp.synalm(cl_phiphi_true, lmax=lmax - 1, new=True).astype(np.complex128)
    phi_start_packed = _alm_hp_to_packed(phi_start_hp, lmax)

    # alm start: MAP (same as production pilot)
    print(f"Finding MAP estimate ({args.map_steps} Adam steps)...")
    t_map = time.time()
    x0 = np.asarray(
        find_map_estimate(model, n_steps=args.map_steps, learning_rate=args.map_lr),
        dtype=np.float64,
    )
    print(f"  MAP done in {time.time() - t_map:.1f}s")

    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    L_arr, _ = _alm_index_lm(lmax, n_real, n_imag)
    ell_bins = [(lo, min(hi, lmax)) for lo, hi in
                [(2, 10), (10, 30), (30, 60), (60, 100), (100, 150)] if lo < lmax]

    all_summaries = {}
    for step_size in step_sizes:
        for L in L_vals:
            tag = f"step{step_size:.4g}_L{L:.4g}"
            out_path = os.path.join(args.out_dir, f"phi_mclmc_{tag}.npz")
            print(f"\n{'='*60}")
            print(f"Running: step_size={step_size}, L={L}, n_steps={args.n_steps}")

            t0 = time.time()
            _samples, phi_samples, _logp, _accepts, _final_step, _cl_pp = run_gibbs_chain(
                model,
                n_samples=args.n_samples,
                n_burnin=args.n_burnin,
                hmc_step_size=0.01,
                n_lfs=10,
                initial_params=x0,
                cl_phiphi_full=cl_phiphi_true,
                phi_initial=phi_start_packed,
                phi_sampler="mclmc",
                phi_hmc_step_size=step_size,
                phi_n_lfs=args.n_steps,
                phi_mclmc_L=L,
                phi_mass_matrix="prior",
                sample_cl_phiphi=True,
                seed=args.seed,
            )
            wall_s = time.time() - t0

            summary = report_config(
                step_size, L, args.n_steps, phi_samples, L_arr, ell_bins,
                wall_s, args.n_samples,
            )
            all_summaries[tag] = summary

            np.savez(
                out_path,
                phi_samples=phi_samples,
                phi_true_packed=phi_true_packed,
                step_size=step_size, L=L, n_steps=args.n_steps,
                wall_s=wall_s,
            )
            print(f"  Saved to {out_path}")

    # Final comparison table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE: configs with no oscillation (no early negative ACF bounce)")
    print(f"  {'config':<22}  {'[2,10) r1':>10}  {'[2,10) 1st-neg':>14}  {'[60,100) r1':>12}")
    for tag, s in all_summaries.items():
        b210 = s.get((2, 10), {})
        b6100 = s.get((60, 100), {})
        r1_210 = b210.get("r1", np.nan)
        fn_210 = b210.get("first_neg_lag", None)
        osc_210 = b210.get("oscillates", False)
        r1_6100 = b6100.get("r1", np.nan)
        flag = " <-- OSCILLATES" if osc_210 else ""
        fn_str = f"lag {fn_210}" if fn_210 else "none"
        print(f"  {tag:<22}  {r1_210:>10.3f}  {fn_str:>14}  {r1_6100:>12.3f}{flag}")

    # Identify best non-oscillating config
    non_osc = [(tag, s) for tag, s in all_summaries.items()
               if not s.get((2, 10), {}).get("oscillates", True)]
    if non_osc:
        # Pick the one with smallest first_below02_lag in worst bin, as proxy for best mixing
        def worst_crossing(tag_s):
            s = tag_s[1]
            crossings = [v.get("first_below02_lag") or 9999 for v in s.values()]
            return max(crossings)
        best_tag, best_s = min(non_osc, key=worst_crossing)
        print(f"\nRECOMMENDED CONFIG (no oscillation, best mixing): {best_tag}")
        print("  -> Use this step_size/L in the next full equilibration pilot run.")
    else:
        print("\nAll configs show oscillation. Try step_size < min(step_sizes) in a follow-up.")

    print(f"\nAll results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
