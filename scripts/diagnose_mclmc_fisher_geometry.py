"""
Bounded geometry-fix spike (ROADMAP.md, 2026-08-14): does a Fisher-informed
(posterior-curvature) phi mass matrix fix MCLMC's equilibration pathology at
lmax=128, where the prior-only mass matrix has now failed the long-run gate
twice (job 11710475 untuned, job 11744132 tuned -- see achievements.md)?

Motivation (lensing.py::estimate_phi_diag_fisher's own docstring, written
2026-08-08 but never tested against MCLMC): "the whitened posterior variance
is highly non-uniform across L (tight where the likelihood dominates at low
L, prior-dominated and looser at high L), so a single global step size
settles at whatever the tightest dimension needs, moving almost nowhere in
the rest." That is exactly MCLMC's observed failure mode (worst lag-1=1.000
and worst drift=6.55 sigma across ALL FIVE l-bins simultaneously in job
11744132, not just one problem bin) -- a single scalar step_size/L cannot
compensate for per-L curvature heterogeneity no matter how it is tuned.

Caveat this script does NOT resolve: the Fisher mass matrix is a one-time
burn-in estimate, frozen thereafter -- incompatible with Block 4
(sample_cl_phiphi=True) resampling cl_phiphi_full every sweep (see
run_gibbs_chain's docstring/guard in samplers.py). So this is run with
Block 4 OFF (cl_phiphi_full fixed at the true LCDM spectrum) -- a diagnostic
of whether preconditioning fixes the geometry at all, not a directly
deployable production configuration. If GO, reconciling with Block 4 is a
separate follow-up (e.g. periodic re-estimation instead of once-at-burn-in).

Reuses scripts/pilot_coverage_equilibration.py's helpers (diagnostics, MAP
start, warm-start-away-from-truth protocol) verbatim -- only the
run_gibbs_chain call and the Block-4-shaped return tuple differ.

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/diagnose_mclmc_fisher_geometry.py \
    --lmax 128 --nside 128 --n_burnin 400 --n_samples 3300 \
    --checkpoint_path results/analysis/diagnose_mclmc_fisher_ckpt.npz
"""
import argparse
import time

import healpy as hp
import numpy as np
import tensorflow as tf
from pilot_coverage_equilibration import (
    DRIFT_NOGO_SIGMA,
    LAG1_NOGO,
    LCDM_PARAMS,
    PHI_ACCEPT_NOGO,
    get_cl_phiphi,
    phi_power_traces,
    report_equilibration,
)

from diffcmb import CosmologyAdvancedSampling, run_gibbs_chain
from diffcmb.alm_utils import packed_sizes
from diffcmb.lensing import _alm_hp_to_packed, compute_sl_phi_np, lens_map_tf
from diffcmb.power import call_CAMB_map
from diffcmb.samplers import _alm_index_lm, find_map_estimate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lmax", type=int, default=128)
    p.add_argument("--nside", type=int, default=128)
    p.add_argument("--noisesig", type=float, default=1.0)
    p.add_argument("--n_burnin", type=int, default=400)
    p.add_argument("--n_samples", type=int, default=3300)
    p.add_argument("--map_steps", type=int, default=2000)
    p.add_argument("--map_lr", type=float, default=0.01)
    p.add_argument("--hmc_step_size", type=float, default=0.01)
    p.add_argument("--n_lfs", type=int, default=10)
    p.add_argument("--phi_hmc_step_size", type=float, default=0.005,
                   help="MCLMC step_size; 0.005 is the nside=128 tuning "
                        "grid's non-oscillating recommendation (job 11717685)")
    p.add_argument("--phi_n_lfs", type=int, default=30)
    p.add_argument("--phi_mclmc_L", type=float, default=100.0)
    p.add_argument("--phi_fisher_warmup_iter", type=int, default=20,
                   help="burn-in sweep at which the one-time Fisher curvature "
                        "estimate is taken (must be < n_burnin)")
    p.add_argument("--phi_fisher_n_probes", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint_path", type=str,
                   default="results/analysis/diagnose_mclmc_fisher_ckpt.npz")
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument("--out", type=str,
                   default="results/analysis/diagnose_mclmc_fisher.npz")
    args = p.parse_args()

    lmax, nside = args.lmax, args.nside

    print(f"=== MCLMC + Fisher-mass-matrix geometry-fix spike: lmax={lmax} "
          f"(nside={nside}) ===")
    print("Warm-started AWAY from truth; Block 4 (C_L^phiphi|phi) OFF "
          "(fisher mass matrix requires a frozen spectrum); "
          "phi_mass_matrix='fisher'.\n")

    print("Building matrix-free-SHT model (synthetic, full-sky)...")
    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=args.noisesig,
        data_mode="synthetic", dtype=tf.complex128, use_matrixfree_sht=True,
    )
    model._ensure_tf_tensors()
    assert len(model.unmasked_idx) == model.NPIX, (
        "pilot assumes full-sky (synthetic) data"
    )

    print("Drawing (alm_true, phi_true) from CAMB spectra at the fixed LCDM cosmology...")
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

    print("Lensing the true sky through the matrix-free forward operator and adding noise...")
    T_lensed_true = lens_map_tf(
        model, tf.constant(alm_true_packed, tf.float64), phi_true_hp
    ).numpy()
    if not np.all(np.isfinite(T_lensed_true)):
        raise RuntimeError("lens_map_tf produced NaN/Inf -- aborting")

    noisy_map = T_lensed_true + rng_noise.normal(0.0, args.noisesig, size=model.NPIX)
    model.prior_map = noisy_map
    model.prior_map_masked = tf.convert_to_tensor(
        noisy_map[model.unmasked_idx], dtype=tf.float64
    )

    np.random.seed(rng_start.integers(0, 2**31 - 1))
    alm_start_hp = hp.synalm(cl_true, lmax=lmax - 1, new=True).astype(np.complex128)
    phi_start_hp = hp.synalm(cl_phiphi_true, lmax=lmax - 1, new=True).astype(np.complex128)
    alm_start_packed = _alm_hp_to_packed(alm_start_hp, lmax)
    phi_start_packed = _alm_hp_to_packed(phi_start_hp, lmax)

    S_phi_true = compute_sl_phi_np(phi_true_packed, lmax)
    S_phi_start = compute_sl_phi_np(phi_start_packed, lmax)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_pow_ratio = np.nanmean(S_phi_start[2:] / S_phi_true[2:])
    start_corr = float(
        np.dot(phi_start_packed, phi_true_packed)
        / (np.linalg.norm(phi_start_packed) * np.linalg.norm(phi_true_packed))
    )
    print(f"  start-vs-truth phi: power ratio={phi_pow_ratio:.3f}, "
          f"cosine similarity={start_corr:+.4f} (should be ~0 -- independent draws)")

    if args.map_steps > 0:
        t_map = time.time()
        x0 = np.asarray(
            find_map_estimate(model, n_steps=args.map_steps,
                              learning_rate=args.map_lr),
            dtype=np.float64,
        )
        print(f"  MAP alm start found in {time.time() - t_map:.1f}s")
        map_alm = x0[lmax - 2:]
        map_corr = float(
            np.dot(map_alm, alm_true_packed)
            / (np.linalg.norm(map_alm) * np.linalg.norm(alm_true_packed))
        )
        print(f"  MAP-vs-truth alm cosine similarity={map_corr:+.4f} "
              f"(expected >0: data-driven, not truth-initialised)")
    else:
        x0 = np.concatenate([np.log(cl_true[2:lmax]), alm_start_packed])

    print(f"\nRunning 3-block Gibbs chain (Block 4 off): n_burnin={args.n_burnin}, "
          f"n_samples={args.n_samples}, n_lfs={args.n_lfs}, "
          f"phi_n_lfs={args.phi_n_lfs}, phi_mass_matrix=fisher "
          f"(warmup_iter={args.phi_fisher_warmup_iter}, "
          f"n_probes={args.phi_fisher_n_probes}) "
          f"(checkpoint={args.checkpoint_path})...")
    t0 = time.time()
    samples, phi_samples, logp, accepts, final_step = run_gibbs_chain(
        model,
        n_samples=args.n_samples,
        n_burnin=args.n_burnin,
        hmc_step_size=args.hmc_step_size,
        n_lfs=args.n_lfs,
        initial_params=x0,
        cl_phiphi_full=cl_phiphi_true,
        phi_initial=phi_start_packed,
        phi_hmc_step_size=args.phi_hmc_step_size,
        phi_n_lfs=args.phi_n_lfs,
        phi_sampler="mclmc",
        phi_mclmc_L=args.phi_mclmc_L,
        phi_mass_matrix="fisher",
        phi_fisher_warmup_iter=args.phi_fisher_warmup_iter,
        phi_fisher_n_probes=args.phi_fisher_n_probes,
        sample_cl_phiphi=False,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
        checkpoint_every=args.checkpoint_every,
    )
    elapsed = time.time() - t0
    n_collected = max(1, len(samples))
    sweeps = args.n_burnin + n_collected
    print(f"  done in {elapsed / 3600:.2f}h "
          f"({elapsed / n_collected:.2f}s/collected-sample)")
    print(f"  alm-block accept rate: {accepts.mean():.3f}")
    print(f"  ~{elapsed / n_collected:.1f}s/sweep at lmax={lmax}")

    if not (np.all(np.isfinite(samples)) and np.all(np.isfinite(phi_samples))
            and np.all(np.isfinite(logp))):
        raise RuntimeError("NaN/Inf in samples -- spike FAILED before diagnostics")

    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    L_arr, _m_arr = _alm_index_lm(lmax, n_real, n_imag)

    ell_bins = [(lo, min(hi, lmax)) for lo, hi in
                [(2, 10), (10, 30), (30, 60), (60, 100), (100, 150)]
                if lo < lmax]

    traces = phi_power_traces(phi_samples, L_arr, ell_bins)
    phi_rows = report_equilibration(traces, "phi block (lensing potential, fisher mass matrix)")

    phi_accept = np.nan
    try:
        _ck = np.load(args.checkpoint_path, allow_pickle=True)
        if "phi_accepts" in _ck.files:
            phi_accept = float(np.mean(_ck["phi_accepts"]))
    except (OSError, ValueError) as exc:
        print(f"  ! could not read phi accept rate from checkpoint: {exc}")
    print(f"\n  phi block accept rate: {phi_accept:.4f} (gate floor is {PHI_ACCEPT_NOGO})")

    lag1 = np.array([r[2] for r in phi_rows], dtype=np.float64)
    drifts = np.array([r[-1] for r in phi_rows], dtype=np.float64)
    worst_lag1 = np.nanmax(lag1) if len(lag1) else np.nan
    worst_drift = np.nanmax(np.abs(drifts)) if len(drifts) else np.nan
    n_nan = int(np.sum(~np.isfinite(lag1)) + np.sum(~np.isfinite(drifts)))

    print("\n=== Verdict (geometry-fix spike, Block 4 OFF -- diagnostic only, "
          "not a production configuration) ===")
    if np.isfinite(phi_accept) and phi_accept < PHI_ACCEPT_NOGO:
        print(f"NO-GO: phi accept/non-divergence rate {phi_accept:.4f} < {PHI_ACCEPT_NOGO}.")
    elif n_nan:
        print(f"NO-GO / INCONCLUSIVE: {n_nan} non-finite diagnostic(s).")
    elif worst_lag1 >= LAG1_NOGO or worst_drift >= DRIFT_NOGO_SIGMA:
        reasons = []
        if worst_lag1 >= LAG1_NOGO:
            reasons.append(f"worst lag-1 autocorrelation {worst_lag1:.3f} >= {LAG1_NOGO}")
        if worst_drift >= DRIFT_NOGO_SIGMA:
            reasons.append(f"worst |drift| {worst_drift:.2f} sigma >= {DRIFT_NOGO_SIGMA}")
        print(f"NO-GO: {'; '.join(reasons)}. Fisher preconditioning does NOT fix MCLMC's "
              f"equilibration pathology at lmax={lmax} in {sweeps} sweeps -- the "
              f"heterogeneous-curvature hypothesis is not the (whole) story, or the "
              f"one-shot burn-in Fisher estimate is too noisy to help (per achievements.md's "
              f"HMC+fisher closure). Trigger 3(b) (schedule threshold / lower lmax) becomes "
              f"the remaining option on the MCLMC track.")
    else:
        print(f"GO: worst lag-1 autocorrelation {worst_lag1:.3f} (< {LAG1_NOGO}) and "
              f"worst |drift| {worst_drift:.2f} sigma (< {DRIFT_NOGO_SIGMA}) -- Fisher "
              f"preconditioning fixes MCLMC's equilibration at lmax={lmax} with Block 4 "
              f"OFF. Follow-up needed before this is production-usable: reconcile with "
              f"Block 4's per-sweep spectrum resampling (e.g. periodic re-estimation of "
              f"the Fisher term instead of once-at-burn-in) -- NOT yet validated here.")

    np.savez(
        args.out,
        alm_samples=samples, phi_samples=phi_samples, logp=logp, accepts=accepts,
        cl_true=cl_true, cl_phiphi_true=cl_phiphi_true,
        alm_true_packed=alm_true_packed, phi_true_packed=phi_true_packed,
        alm_start_packed=alm_start_packed, phi_start_packed=phi_start_packed,
        start_cosine_similarity=start_corr,
        phi_traces=np.array([traces[k] for k in traces], dtype=np.float64),
        phi_trace_bins=np.array(list(traces.keys()), dtype=np.int64),
        phi_rows=np.array(phi_rows, dtype=np.float64),
        seconds_total=elapsed, seconds_per_sweep=elapsed / n_collected,
        lmax=lmax, nside=nside, phi_n_lfs=args.phi_n_lfs,
    )
    print(f"\nSaved chain + traces + equilibration stats to {args.out}")


if __name__ == "__main__":
    main()
