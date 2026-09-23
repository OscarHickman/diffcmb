"""
ROADMAP.md Section 1, Priority 1: the *prerequisite gate* for the
multi-realization rank/coverage test.

This is deliberately NOT the coverage test. It is the single pilot chain that
decides whether the O(10-20)-chain ensemble is worth launching at all, and it
answers exactly one question:

    at this lmax, does the phi block equilibrate within an affordable
    number of sweeps when warm-started AWAY from truth?

Why this gate exists (achievements.md, and the ROADMAP entry): a rank/coverage
test on a non-equilibrated chain produces confidently wrong uniformity plots --
strictly worse than no test. This project has already been burned twice by that
failure mode. Job 11612969's phi block never equilibrated at lmax=300, and --
critically -- *both* available mixing estimators (the Sokal IAT estimator and
`integrated_autocorr_time`'s zero-variance guard) reported near-perfect mixing
on those non-mixing chains.

So this script deliberately uses NO IAT/ESS estimator anywhere. The diagnostics
are the two that actually caught the failure:

  1. Direct lag-k autocorrelation of the binned phi power trace, at several
     explicit lags. No windowing, no automatic truncation, no variance guard --
     just r_k on the raw series.
  2. A monotonic-drift test: the signature of the 11612969 failure was a phi
     power trace still sliding steadily toward the truth at the end of the
     window, rather than scattering about a stationary level. Comparing the
     first- and last-third means against the within-third scatter detects that
     directly, and (unlike an autocorrelation) it cannot be fooled by a smooth
     slow ramp.

Two differences from scripts/validate_sim_lmax300_lensing.py are intentional
and are the whole point:

  * **Not warm-started at truth.** That script warm-starts *at* the truth to
    isolate conditional recovery from burn-in time. A coverage test needs the
    opposite: a chain that starts at the truth produces a rank statistic that
    inherits its initialisation. Here the alm block starts from the *data-driven
    MAP* and phi from an independent prior draw (a separate seed stream from the
    truth), so nothing about the truth leaks into the start.

    The MAP start is deliberate and is NOT a truth leak: find_map_estimate
    minimises psi over the data alone and never sees alm_true/phi_true, so it is
    legitimate for a calibration test in a way a truth start is not. It is also
    what production chains do (scripts/run_sampler.py, and this codebase's
    standing answer to burn-in-to-mode time).

    !! Job 11663105 (2026-07-30) is why the MAP start is mandatory here. That
    run used a cold independent prior draw for alm with only 100 burn-in sweeps
    and no MAP pre-solve, and the phi block froze completely: phi accept 0.004
    (0.000 over the last 50 sweeps), binned phi power at L=2-6 bit-identical
    across sweeps 0/50/150/300, and the frozen phi state carrying realized
    C_L=2 ~ 2.7e-06 against a fiducial 1.1e-08 (~240x). Mechanism: a badly wrong
    alm leaves a huge residual, the phi block inflates to absorb it, Block 4
    faithfully fits that inflated phi (its draws tracked the state's realized
    power to within the normal inverse-Gamma spread -- Block 4 is not at fault),
    the phi mass matrix is rebuilt from the inflated spectrum, and phi locks at
    ~0 accept. Fixing the alm start attacks that root cause; the phi accept gate
    below is the tripwire if it recurs.
  * **Block 4 on** (`sample_cl_phiphi=True`), which the ensemble requires for
    the C_L^phiphi rank statistic to exist at all. That forces
    `phi_mass_matrix='prior'` -- the two are mutually exclusive by design, and
    the fisher route is closed as unfavorable anyway (achievements.md).

The cost-scaling measurement the ROADMAP asks for falls out of the same run:
seconds/sweep is reported and saved, which is what sizes the ensemble.

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/pilot_coverage_equilibration.py \
    --lmax 128 --nside 128 --n_burnin 400 --n_samples 600 \
    --map_steps 500 --checkpoint_path results/analysis/pilot_coverage_lmax128_mclmc_ckpt.npz

Block 3 defaults to phi_sampler='mclmc' (2026-08-08, gate job 11708844 GO --
see achievements.md). Pass --phi_sampler hmc --phi_hmc_step_size 0.05
--phi_n_lfs 240 to fall back to the retired HMC path.
"""
import argparse
import time

import healpy as hp
import numpy as np
import tensorflow as tf

from diffcmb import CosmologyAdvancedSampling, run_gibbs_chain
from diffcmb.alm_utils import packed_sizes
from diffcmb.lensing import _alm_hp_to_packed, compute_sl_phi_np, lens_map_tf
from diffcmb.power import call_CAMB_map, fiducial_spectra
from diffcmb.samplers import _alm_index_lm, find_map_estimate

LCDM_PARAMS = [67.74, 0.0486, 0.2589, 0.06, 0.0, 0.066]

# Lags at which the raw autocorrelation of the phi power trace is reported.
# Chosen to span "adjacent sweeps" through "a good fraction of the window" so a
# slowly-decaying series is visible as a flat profile rather than a decaying one.
# Extended past the original (1,5,10,25,50) after job 11663477 (2026-07-30):
# that run's lag-1 alone triggered NO-GO (0.981 >= LAG1_NOGO), but a post-hoc
# re-analysis of the saved trace out to lag=400 showed real decay through zero
# by lag~75-100 in every l-bin -- qualitatively different from job 11612969's
# genuinely-stuck 0.9999-forever, no-decay profile. The wider table makes that
# distinction visible in the run's own printed output instead of requiring a
# separate offline check every time. This does NOT change the GO/NO-GO
# threshold logic below (still keyed on lag-1 and drift only, deliberately --
# see the verdict block) -- it only makes the fuller picture auditable.
DIAGNOSTIC_LAGS = (1, 5, 10, 25, 50, 75, 100, 150, 200)

# A lag-1 autocorrelation at or above this is treated as not-yet-mixing. The
# toy-scale leapfrog fix reached 0.576 (achievements.md); anything approaching
# the 0.9999 of the failed job 11612969 is the pathology this gate must catch.
LAG1_NOGO = 0.90

# Drift is flagged when |mean(last third) - mean(first third)| exceeds this
# many within-third standard deviations. At 2.0 a genuinely stationary noisy
# trace passes comfortably while 11612969's steady ramp would not.
DRIFT_NOGO_SIGMA = 2.0

# Hard floor on the phi block's accept rate. Job 11663105 froze at 0.004 while
# the alm block still reported a plausible-looking step size, so a low phi
# accept rate is the earliest and least ambiguous signal that the run is dead --
# it needs its own gate rather than being inferred from the trace statistics.
PHI_ACCEPT_NOGO = 0.15


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
    """Plain lag-k Pearson autocorrelation of a 1-D trace.

    Deliberately hand-rolled rather than routed through any IAT/ESS helper:
    both estimators in this codebase reported near-perfect mixing on the
    non-mixing phi chains of job 11612969 (achievements.md), so this gate must
    not depend on them. Returns NaN when the series is too short or constant,
    and NaN is treated as a NO-GO by the caller rather than as a pass.
    """
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


def drift_sigma(series):
    """Signed first-third-to-last-third shift, in within-third sigmas.

    This is the diagnostic that caught the job 11612969 failure: a phi power
    trace still sliding monotonically toward the truth at the end of the
    sampling window. Positive means the trace rose over the run.
    """
    n = len(series)
    if n < 9:
        return np.nan
    third = n // 3
    first, last = series[:third], series[-third:]
    scatter = np.sqrt(0.5 * (first.var(ddof=1) + last.var(ddof=1)))
    if scatter <= 0:
        return np.nan
    return float((last.mean() - first.mean()) / scatter)


def phi_power_traces(phi_samples, L_arr, ell_bins):
    """Per-sweep bin-averaged phi power: {(lo,hi): trace of length n_samples}."""
    power = phi_samples ** 2
    traces = {}
    for lo, hi in ell_bins:
        mask = (L_arr >= lo) & (L_arr < hi)
        if mask.any():
            traces[(lo, hi)] = power[:, mask].mean(axis=1)
    return traces


def report_equilibration(traces, label):
    """Print the lag-k and drift table for a set of traces; return per-bin rows."""
    print(f"\n--- {label}: phi equilibration (direct lag-k autocorrelation + "
          f"drift; no IAT/ESS estimator used) ---")
    header = "  l-bin        " + "  ".join(f"r_{k:<4d}" for k in DIAGNOSTIC_LAGS) + "   drift_sigma"
    print(header)
    rows = []
    for (lo, hi), trace in traces.items():
        acs = [lag_autocorr(trace, k) for k in DIAGNOSTIC_LAGS]
        d = drift_sigma(trace)
        ac_str = "  ".join(f"{a:6.3f}" if np.isfinite(a) else "   nan" for a in acs)
        print(f"  [{lo:4d},{hi:4d})  {ac_str}   {d:8.2f}")
        rows.append((lo, hi, *acs, d))

    # Reporting only -- does not feed the GO/NO-GO verdict. First lag (among
    # DIAGNOSTIC_LAGS) at which |r_k| drops under 0.2: a rough thinning-interval
    # estimate for sizing the coverage ensemble's --thin, and a way to tell a
    # genuinely-decaying-but-slow trace (this crosses somewhere in the table)
    # from job 11612969's profile (stayed >0.99 at every probed lag, so this
    # would print "> max lag" for every bin).
    print("  first lag with |r_k| < 0.2 (thinning-interval estimate, not a gate):")
    for (lo, hi), trace in traces.items():
        acs = [lag_autocorr(trace, k) for k in DIAGNOSTIC_LAGS]
        crossing = next(
            (k for k, a in zip(DIAGNOSTIC_LAGS, acs) if np.isfinite(a) and abs(a) < 0.2),
            None,
        )
        label_str = f"lag {crossing}" if crossing is not None else f"> {DIAGNOSTIC_LAGS[-1]}"
        print(f"    [{lo:4d},{hi:4d})  {label_str}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lmax", type=int, default=128)
    p.add_argument("--nside", type=int, default=128)
    p.add_argument("--noisesig", type=float, default=1.0)
    p.add_argument("--lensing_operator", choices=("exact", "bilinear"), default="exact",
                   help="Forward lensing operator. 'exact' (default since 2026-09-23) "
                        "evaluates the alm at the deflected positions and lenses by "
                        "+grad(phi); 'bilinear' is the legacy interpolation (and "
                        "-grad(phi) sign) that every earlier chain used. ROADMAP D3.")
    p.add_argument("--fiducial", choices=("corrected", "legacy"), default="corrected",
                   help="'corrected' = power.fiducial_spectra (physical densities, "
                        "raw C_l, unlensed TT); 'legacy' = the pre-2026-09-23 "
                        "call_CAMB_map spectra, for reproducing old chains only.")
    p.add_argument("--n_burnin", type=int, default=400,
                   help="raised from 100 after job 11663105's cold-start freeze")
    p.add_argument("--n_samples", type=int, default=600)
    # 2000 steps at lr=0.01 are the *validated production* MAP settings recorded
    # in achievements.md (the burn-in false-alarm entry: those settings matched
    # the Phase 0 reference C_l to 1-3%). find_map_estimate's own function
    # defaults (500, 2e-4) are far too weak -- at lmax=24 they moved psi by only
    # ~1% and left MAP-vs-truth alm cosine similarity at -0.04, i.e. no better
    # than the cold start that froze job 11663105. Do not lower these.
    p.add_argument("--map_steps", type=int, default=2000,
                   help="Adam steps for the data-driven MAP alm start; 0 falls "
                        "back to a cold prior draw (the configuration that froze "
                        "in job 11663105 -- not recommended)")
    p.add_argument("--map_lr", type=float, default=0.01)
    p.add_argument("--hmc_step_size", type=float, default=0.01)
    p.add_argument("--n_lfs", type=int, default=10)
    p.add_argument("--phi_hmc_step_size", type=float, default=0.1,
                   help="also the MCLMC step_size when --phi_sampler=mclmc; "
                        "0.1 is the winning value from gate job 11708844")
    p.add_argument("--phi_n_lfs", type=int, default=30,
                   help="HMC trajectory length, or MCLMC n_steps when "
                        "--phi_sampler=mclmc; 30 is the winning value from "
                        "gate job 11708844 (was 80/240 for the retired HMC path)")
    p.add_argument("--phi_sampler", type=str, default="mclmc",
                   choices=("hmc", "mclmc", "nuts"),
                   help="Block 3 integrator. 'mclmc' promoted to default "
                        "2026-08-08: gate job 11708844 showed MCLMC beating "
                        "HMC ESS/wall-clock-second on all 6 probed l-bins at "
                        "lmax=128 (achievements.md). 'nuts' (2026-08-17 "
                        "spike): dynamic trajectory length via the no-U-turn "
                        "criterion, since every fixed-step-size lever (HMC "
                        "and MCLMC alike) has failed the equilibration gate "
                        "regardless of tuning -- ignores --phi_n_lfs.")
    p.add_argument("--phi_mclmc_L", type=float, default=200.0,
                   help="MCLMC momentum decoherence length; 200 is the "
                        "winning value from gate job 11708844 at lmax=128")
    p.add_argument("--phi_nuts_max_tree_depth", type=int, default=10,
                   help="only used when --phi_sampler=nuts")
    p.add_argument("--phi_mass_matrix", type=str, default="prior",
                   choices=("prior", "fisher", "block"),
                   help="Block 3 HMC/NUTS/MCLMC preconditioner. 'prior' "
                        "(default, unchanged) is diagonal-in-L, prior-curvature "
                        "only -- what every gate run to date has used, together "
                        "with Block 4 (C_L^phiphi|phi) forced on. 'block' "
                        "(2026-08-18, ROADMAP.md/achievements.md) adds "
                        "lensing.py::estimate_phi_block_hessian's cross-L "
                        "Nystrom correction, the structurally motivated fix for "
                        "the cross-L Hessian coupling "
                        "diagnose_phi_hessian_coupling.py found significant -- "
                        "no diagonal mass matrix ('prior' or 'fisher') can "
                        "represent it. 'fisher' adds only the diagonal "
                        "likelihood-curvature term (samplers.py::"
                        "build_phi_posterior_mass_sqrt), kept for comparison. "
                        "'fisher' is a one-time burn-in estimate, frozen "
                        "thereafter, and (samplers.py) not supported together "
                        "with Block 4 -- selecting it FORCES Block 4 off for "
                        "this run (sample_cl_phiphi=False, cl_phiphi_true used "
                        "as a fixed spectrum throughout). 'block' IS supported "
                        "with Block 4 (2026-08-23, ROADMAP.md/samplers.py): its "
                        "expensive likelihood-curvature estimate is frozen the "
                        "same way, but the diagonal prior-precision term is "
                        "cheaply rebuilt every sweep to track Block 4's "
                        "resampled spectrum, so Block 4 stays ON by default "
                        "for 'block' same as 'prior'.")
    p.add_argument("--phi_fisher_warmup_iter", type=int, default=20,
                   help="burn-in sweep at which the 'fisher'/'block' mass "
                        "matrix is computed (once, then frozen)")
    p.add_argument("--phi_fisher_n_probes", type=int, default=8,
                   help="coordinate samples per L for the diagonal "
                        "Fisher/prior-curvature term in 'fisher'/'block' mode")
    p.add_argument("--phi_block_n_probes", type=int, default=6,
                   help="coordinate samples per m-block for 'block' mode's "
                        "cross-L Nystrom correction (ignored otherwise)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint_path", type=str,
                   default="results/analysis/pilot_coverage_lmax128_ckpt.npz")
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument("--out", type=str,
                   default="results/analysis/pilot_coverage_lmax128.npz")
    args = p.parse_args()

    lmax, nside = args.lmax, args.nside

    # 'fisher' is a one-time burn-in estimate, frozen thereafter (samplers.py),
    # not supported together with Block 4's every-sweep spectrum resampling.
    # 'block' IS supported (2026-08-23) -- see --phi_mass_matrix's help text.
    use_block4 = args.phi_mass_matrix in ('prior', 'block')

    print(f"=== Coverage-test prerequisite gate: phi equilibration pilot at "
          f"lmax={lmax} (nside={nside}) ===")
    print(f"Warm-started AWAY from truth; Block 4 (C_L^phiphi|phi) "
          f"{'ON' if use_block4 else 'OFF (forced by phi_mass_matrix)'}; "
          f"phi_mass_matrix={args.phi_mass_matrix!r}.\n")

    print("Building matrix-free-SHT model (synthetic, full-sky)...")
    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=args.noisesig,
        data_mode="synthetic", dtype=tf.complex128, use_matrixfree_sht=True,
        lensing_operator=args.lensing_operator,
    )
    model._ensure_tf_tensors()
    assert len(model.unmasked_idx) == model.NPIX, (
        "pilot assumes full-sky (synthetic) data"
    )

    print("Drawing (alm_true, phi_true) from CAMB spectra at the fixed LCDM cosmology...")
    if args.fiducial == "corrected":
        cl_true, cl_phiphi_true = fiducial_spectra(lmax)
    else:
        cl_true = call_CAMB_map(LCDM_PARAMS, lmax)
        cl_phiphi_true = get_cl_phiphi(lmax)

    # Truth stream and start-point stream are deliberately independent: the
    # start must not be a perturbation of the truth, or the chain begins
    # already inside the posterior and the gate proves nothing.
    rng_truth = np.random.default_rng(args.seed)
    rng_start = np.random.default_rng(args.seed + 10_000)
    rng_noise = np.random.default_rng(args.seed + 20_000)

    # hp.synalm draws from numpy's global state, so the two independent streams
    # are realised by seeding it from each generator in turn.
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
        raise RuntimeError("lens_map_tf produced NaN/Inf -- aborting pilot")

    noisy_map = T_lensed_true + rng_noise.normal(0.0, args.noisesig, size=model.NPIX)
    model.prior_map = noisy_map
    model.prior_map_masked = tf.convert_to_tensor(
        noisy_map[model.unmasked_idx], dtype=tf.float64
    )

    # --- The warm-start-away-from-truth initial state ---
    np.random.seed(rng_start.integers(0, 2**31 - 1))
    alm_start_hp = hp.synalm(cl_true, lmax=lmax - 1, new=True).astype(np.complex128)
    phi_start_hp = hp.synalm(cl_phiphi_true, lmax=lmax - 1, new=True).astype(np.complex128)
    alm_start_packed = _alm_hp_to_packed(alm_start_hp, lmax)
    phi_start_packed = _alm_hp_to_packed(phi_start_hp, lmax)

    # Report how far the start actually is, so "away from truth" is a measured
    # fact in the log rather than an assumption.
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

    # --- alm start: the data-driven MAP, not the cold prior draw ---
    # Must come AFTER the data is installed on the model above, since psi (and
    # hence the MAP) depends on it. find_map_estimate ignores alm_start_packed
    # and descends from model.prior_parameters_tf() using the data alone, so the
    # truth never enters. It optimises the *unlensed* posterior (_psi_tf_raw),
    # which at nonzero phi is an approximation -- fine for a starting point, and
    # vastly better than the cold draw that froze job 11663105.
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
        # Unlike the phi cosine similarity this one SHOULD be well above zero:
        # the MAP is data-driven and the data contain the truth. That is not a
        # leak -- it is the posterior doing its job. Reported so the distinction
        # between "informed by data" and "initialised at truth" stays auditable.
        print(f"  MAP-vs-truth alm cosine similarity={map_corr:+.4f} "
              f"(expected >0: data-driven, not truth-initialised)")
    else:
        print("  ! map_steps=0: cold prior-draw alm start -- this is the "
              "configuration that froze in job 11663105")
        x0 = np.concatenate([np.log(cl_true[2:lmax]), alm_start_packed])

    print(f"\nRunning 4-block Gibbs chain: n_burnin={args.n_burnin}, "
          f"n_samples={args.n_samples}, n_lfs={args.n_lfs}, "
          f"phi_n_lfs={args.phi_n_lfs} "
          f"(checkpoint={args.checkpoint_path})...")
    t0 = time.time()
    gibbs_kwargs = {
        "n_samples": args.n_samples,
        "n_burnin": args.n_burnin,
        "hmc_step_size": args.hmc_step_size,
        "n_lfs": args.n_lfs,
        "initial_params": x0,
        "cl_phiphi_full": cl_phiphi_true,
        "phi_initial": phi_start_packed,
        "phi_hmc_step_size": args.phi_hmc_step_size,
        "phi_n_lfs": args.phi_n_lfs,
        "phi_sampler": args.phi_sampler,
        "phi_mclmc_L": args.phi_mclmc_L,
        "phi_nuts_max_tree_depth": args.phi_nuts_max_tree_depth,
        "phi_mass_matrix": args.phi_mass_matrix,
        "seed": args.seed,
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_every": args.checkpoint_every,
    }
    if args.phi_mass_matrix in ('fisher', 'block'):
        gibbs_kwargs.update(
            phi_fisher_warmup_iter=args.phi_fisher_warmup_iter,
            phi_fisher_n_probes=args.phi_fisher_n_probes,
        )
    if args.phi_mass_matrix == 'block':
        gibbs_kwargs.update(phi_block_n_probes=args.phi_block_n_probes)
    if use_block4:
        gibbs_kwargs.update(sample_cl_phiphi=True)
        samples, phi_samples, logp, accepts, final_step, cl_phiphi_samples = run_gibbs_chain(
            model, **gibbs_kwargs
        )
    else:
        samples, phi_samples, logp, accepts, final_step = run_gibbs_chain(
            model, **gibbs_kwargs
        )
        # No Block 4: cl_phiphi_full stays fixed at the truth spectrum
        # throughout, so there is no per-sweep sample trace to report --
        # save/report cl_true's own value at every collected sample for a
        # shape-compatible artifact (see np.savez below).
        cl_phiphi_samples = np.tile(np.log(cl_phiphi_true[2:lmax]), (len(samples), 1))
    elapsed = time.time() - t0
    n_collected = max(1, len(samples))
    sweeps = args.n_burnin + n_collected
    print(f"  done in {elapsed / 3600:.2f}h "
          f"({elapsed / n_collected:.2f}s/collected-sample)")
    print(f"  alm-block accept rate: {accepts.mean():.3f}")

    # The cost-scaling number that sizes the ensemble. Note this is only exact
    # for a single uninterrupted submission; on a checkpoint resume `elapsed`
    # covers just this segment, so cross-check against checkpoint mtimes.
    print(f"  ~{elapsed / n_collected:.1f}s/sweep at lmax={lmax} "
          f"(vs ~185s/sweep at lmax=300/phi_n_lfs=80) -- for ensemble sizing")

    if not (np.all(np.isfinite(samples)) and np.all(np.isfinite(phi_samples))
            and np.all(np.isfinite(logp))
            and np.all(np.isfinite(cl_phiphi_samples))):
        raise RuntimeError("NaN/Inf in samples -- pilot FAILED before diagnostics")

    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    L_arr, _m_arr = _alm_index_lm(lmax, n_real, n_imag)

    ell_bins = [(lo, min(hi, lmax)) for lo, hi in
                [(2, 10), (10, 30), (30, 60), (60, 100), (100, 150)]
                if lo < lmax]

    traces = phi_power_traces(phi_samples, L_arr, ell_bins)
    phi_rows = report_equilibration(traces, "phi block (lensing potential)")

    print("\n  Reference: the failed job 11612969 showed lag-1/lag-5 > 0.9999; the "
          "toy-scale\n  leapfrog fix reached mean lag-1 0.576 (achievements.md).")

    # phi's own accept rate is not in run_gibbs_chain's return tuple, but it is
    # written to the checkpoint every checkpoint_every sweeps -- read it back,
    # since a frozen phi block is the failure this gate most needs to catch.
    phi_accept = np.nan
    try:
        _ck = np.load(args.checkpoint_path, allow_pickle=True)
        if "phi_accepts" in _ck.files:
            phi_accept = float(np.mean(_ck["phi_accepts"]))
    except (OSError, ValueError) as exc:
        print(f"  ! could not read phi accept rate from checkpoint: {exc}")
    print(f"\n  phi block accept rate: {phi_accept:.4f} "
          f"(job 11663105 froze at 0.004; gate floor is {PHI_ACCEPT_NOGO})")

    # --- Verdict: GO/NO-GO on launching the O(10-20)-chain ensemble ---
    lag1 = np.array([r[2] for r in phi_rows], dtype=np.float64)
    drifts = np.array([r[-1] for r in phi_rows], dtype=np.float64)
    worst_lag1 = np.nanmax(lag1) if len(lag1) else np.nan
    worst_drift = np.nanmax(np.abs(drifts)) if len(drifts) else np.nan
    n_nan = int(np.sum(~np.isfinite(lag1)) + np.sum(~np.isfinite(drifts)))

    print("\n=== Verdict (prerequisite gate for the coverage ensemble) ===")
    if np.isfinite(phi_accept) and phi_accept < PHI_ACCEPT_NOGO:
        print(f"NO-GO: phi accept rate {phi_accept:.4f} < {PHI_ACCEPT_NOGO}. The phi "
              f"block is not moving, so every trace statistic below is measuring a "
              f"frozen chain rather than a slow one -- this is the job 11663105 "
              f"failure mode. Check first whether the alm start is sane (MAP-vs-truth "
              f"cosine similarity above) and whether the sampled C_L^phiphi has run "
              f"away at low L relative to the fiducial; a phi state carrying inflated "
              f"low-L power rebuilds the mass matrix wrongly and locks the block. Do "
              f"NOT tune phi_n_lfs before ruling that out -- it is not a mixing "
              f"problem when accept is this low.")
    elif n_nan:
        print(f"NO-GO / INCONCLUSIVE: {n_nan} non-finite diagnostic(s) -- a trace was "
              f"constant or too short. A collapsed trace is exactly the "
              f"zero-variance pathology that fooled the IAT estimators before, so "
              f"this is treated as a failure, not a pass. Inspect the table above.")
    elif worst_lag1 >= LAG1_NOGO or worst_drift >= DRIFT_NOGO_SIGMA:
        reasons = []
        if worst_lag1 >= LAG1_NOGO:
            reasons.append(f"worst lag-1 autocorrelation {worst_lag1:.3f} >= {LAG1_NOGO}")
        if worst_drift >= DRIFT_NOGO_SIGMA:
            reasons.append(f"worst |drift| {worst_drift:.2f} sigma >= {DRIFT_NOGO_SIGMA}")
        print(f"NO-GO: {'; '.join(reasons)}. The phi block has not equilibrated in "
              f"{sweeps} sweeps at lmax={lmax}. Do NOT launch the coverage ensemble on "
              f"this configuration -- a rank test here would produce confidently wrong "
              f"uniformity plots. Options in preference order: drop lmax further, "
              f"lengthen the window, or raise phi_n_lfs (the parameter that actually "
              f"controls phi mixing -- achievements.md's leapfrog-schedule entry).")
    else:
        print(f"GO: worst lag-1 autocorrelation {worst_lag1:.3f} (< {LAG1_NOGO}) and "
              f"worst |drift| {worst_drift:.2f} sigma (< {DRIFT_NOGO_SIGMA}) across all "
              f"phi bins -- the trace scatters about a stationary level rather than "
              f"sliding toward truth. lmax={lmax} at ~{elapsed / n_collected:.0f}s/sweep "
              f"is a defensible configuration for the O(10-20)-chain rank/coverage "
              f"ensemble. NOTE: this is an equilibration gate only -- it makes no "
              f"exactness claim, which is what the ensemble itself is for.")

    np.savez(
        args.out,
        alm_samples=samples, phi_samples=phi_samples, logp=logp, accepts=accepts,
        cl_phiphi_samples=cl_phiphi_samples,
        cl_true=cl_true, cl_phiphi_true=cl_phiphi_true,
        alm_true_packed=alm_true_packed, phi_true_packed=phi_true_packed,
        alm_start_packed=alm_start_packed, phi_start_packed=phi_start_packed,
        start_cosine_similarity=start_corr,
        phi_traces=np.array([traces[k] for k in traces], dtype=np.float64),
        phi_trace_bins=np.array(list(traces.keys()), dtype=np.int64),
        phi_rows=np.array(phi_rows, dtype=np.float64),
        diagnostic_lags=np.array(DIAGNOSTIC_LAGS, dtype=np.int64),
        seconds_total=elapsed, seconds_per_sweep=elapsed / n_collected,
        lmax=lmax, nside=nside, phi_n_lfs=args.phi_n_lfs,
        lensing_operator=args.lensing_operator, fiducial=args.fiducial,
        noisesig=args.noisesig,
    )
    print(f"\nSaved chain + traces + equilibration stats to {args.out}")


if __name__ == "__main__":
    main()
