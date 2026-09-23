"""
ROADMAP.md Section 1, Priority 1: one realization of the multi-realization
rank/coverage ensemble. Run as a SLURM array task (one task per realization);
`scripts/aggregate_coverage_ranks.py` pools the outputs into the uniformity
assessment.

Each task draws its OWN independent (alm_true, phi_true, noise) triple from the
fiducial spectra, seeded off `--realization`, runs a full 4-block chain
warm-started away from truth, and saves the posterior samples plus the truth.
Deliberately computes no verdict: all rank statistics and all uniformity
testing live in the aggregation step, so the per-bin statistic can be changed
without re-running 10-20 chains.

Prerequisite: scripts/pilot_coverage_equilibration.py must have returned GO at
this (lmax, phi_n_lfs, n_samples) configuration. A rank test on chains that
have not equilibrated produces confidently wrong uniformity plots -- see that
script's header and achievements.md.

!! READ BEFORE INTERPRETING THE OUTPUT !!
The strict simulation-based-calibration recipe is: draw theta_true from the
prior, simulate data, then check that the rank of theta_true in the posterior
is uniform. That recipe is *not* fully available for the spectrum blocks here,
and the reason is structural rather than an implementation shortcut:

  Block 1's exact conditional is C_l|alm ~ InvGamma(l-0.5, S_l/2). Matching
  that against the C_l likelihood, which goes as C_l^{-(2l+1)/2} exp(-S_l/2C_l)
  = C_l^{-l-0.5} exp(-S_l/2C_l), shows the InvGamma(l-0.5, S_l/2) density
  C_l^{-l-0.5} exp(-S_l/2C_l) *is* the likelihood -- i.e. the implied prior on
  C_l is flat and improper. Block 4 has the same structure for C_L^phiphi.
  An improper prior cannot be drawn from, so there is no way to generate
  C_l_true ~ p(C_l) and the spectra cannot carry a strict SBC rank.

What this script therefore generates *by default* is the *conditional*
ensemble: spectra held at the fiducial values, with alm_true ~ N(0, C_l^fid) and
phi_true ~ N(0, C_L^phiphi,fid) drawn from their exactly-known Gaussian priors.

EXCEPTION, and the way to get a genuinely strict SBC rank for phi:
`--cl_phiphi_prior_nu NU` puts a proper conjugate InvGamma(NU/2, NU*C_L^fid/2)
prior on C_L^phiphi, which makes the target proper in the phi amplitude (the
default flat prior leaves the phi marginal flat in S_L -- improper and rising
with amplitude, which is why the Block-4-ON phi rank in job 11899585 could not
be read as calibration evidence). In that mode the truth is drawn from the SAME
joint prior -- C_L ~ InvGamma, then phi ~ N(0, C_L) -- so generative process and
sampler target agree and the phi rank is a valid SBC statistic. C_l^TT is
untouched and its row remains interval coverage only.
The field-level ranks (alm, phi) are the statistic this design supports most
directly. The spectrum-level ranks are still computed by the aggregator, but
they are an interval-coverage check against the realized power, not a strict
SBC rank, and must be labelled as such in any figure or claim. See the
aggregator's header for exactly what each reported number does and does not
establish.

Usage (single task):
  PYTHONPATH=diffcmb .venv/bin/python scripts/coverage_ensemble_chain.py \
      --realization 0 --lmax 128 --nside 128 \
      --n_burnin 100 --n_samples 600 --phi_n_lfs 80
"""
import argparse
import os
import time

import healpy as hp
import numpy as np
import tensorflow as tf

from diffcmb import CosmologyAdvancedSampling, run_gibbs_chain
from diffcmb.lensing import _alm_hp_to_packed, lens_map_tf
from diffcmb.power import call_CAMB_map, fiducial_spectra
from diffcmb.samplers import PACKING_VERSION, find_map_estimate

LCDM_PARAMS = [67.74, 0.0486, 0.2589, 0.06, 0.0, 0.066]

# Seed-stream offsets. Each realization gets a disjoint block so that the truth,
# the (independent) chain start, and the noise never share a stream -- a shared
# stream would correlate the start with the truth and quietly bias the ranks
# toward the centre.
_STREAM_TRUTH = 1_000_000
_STREAM_START = 2_000_000
_STREAM_NOISE = 3_000_000
# Separate stream for the C_L^phiphi prior draw, so turning the proper prior on
# does not shift the alm/phi/noise streams and change every other realization.
_STREAM_CLPP_PRIOR = 4_000_000


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


def _synalm_pair(cl_tt, cl_pp, lmax, seed):
    """Draw (alm, phi) from their Gaussian priors under a named seed.

    hp.synalm consumes numpy's global RNG state, so the stream is selected by
    seeding it explicitly rather than by passing a Generator.
    """
    np.random.seed(seed)
    alm_hp = hp.synalm(cl_tt, lmax=lmax - 1, new=True).astype(np.complex128)
    phi_hp = hp.synalm(cl_pp, lmax=lmax - 1, new=True).astype(np.complex128)
    return alm_hp, phi_hp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--realization", type=int, required=True,
                   help="realization index; sets all three seed streams")
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
    # 2000 steps at lr=0.01 are the validated production MAP settings, copied
    # from pilot_coverage_equilibration.py so the ensemble and the gate that
    # clears it run the SAME initialisation. map_steps=0 reproduces the old
    # (broken) cold start and is kept only for deliberate A/B testing.
    p.add_argument("--map_steps", type=int, default=2000,
                   help="Adam steps for the data-driven MAP alm start; 0 falls "
                        "back to the cold prior draw that broke job 11887897")
    p.add_argument("--map_lr", type=float, default=0.01)
    # Matches the pilot's burn-in. The original 100 was never gated: the gate
    # ran on the pilot, which burns in 400 sweeps from a MAP start.
    p.add_argument("--n_burnin", type=int, default=400)
    p.add_argument("--n_samples", type=int, default=600)
    p.add_argument("--hmc_step_size", type=float, default=0.01)
    p.add_argument("--n_lfs", type=int, default=10)
    p.add_argument("--phi_hmc_step_size", type=float, default=0.001)
    p.add_argument("--phi_n_lfs", type=int, default=80)
    p.add_argument("--outdir", type=str,
                   default="results/analysis/coverage_ensemble")
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument("--phi_mass_matrix", type=str, default="prior",
                   choices=("prior", "fisher", "block"),
                   help="Block 3 preconditioner. 'block' (per-m Nystrom cross-L "
                        "correction) is the only setting that has ever cleared "
                        "the equilibration gate -- job 11781626, lmax=64, "
                        "Block 4 OFF (achievements.md).")
    p.add_argument("--phi_block_n_probes", type=int, default=6,
                   help="Only used with --phi_mass_matrix block: number of "
                        "single-coordinate FD probes per (channel, m) block "
                        "for the Nystrom curvature estimate. Default 6 matches "
                        "the value 'block' has always been tested at (job "
                        "11874976 NO-GO, achievements.md); raised here to test "
                        "the untested rank-deficiency hypothesis -- the m=0..9 "
                        "blocks that carry all L<10 coordinates are the "
                        "largest and worst-approximated at a fixed rank.")
    p.add_argument("--cl_phiphi_prior_nu", type=float, default=None,
                   help="Put a PROPER conjugate InvGamma(nu/2, nu*C_L^fid/2) "
                        "prior on C_L^phiphi instead of the default flat "
                        "improper one (requires Block 4; nu>2). When set, the "
                        "TRUTH is drawn from that same joint prior "
                        "(C_L ~ InvGamma, then phi ~ N(0,C_L)) so the phi rank "
                        "is a valid SBC test of the proper-prior target.")
    p.add_argument("--no_sample_cl_phiphi", action="store_true",
                   help="Disable Block 4 (C_L^phiphi|phi). Required for the "
                        "lmax=64 GO configuration: Block 4 ON degrades phi "
                        "mixing to lag-1 0.945 vs 0.557 OFF (job 11836793).")
    p.add_argument("--phi_amplitude", type=float, default=1.0,
                   help="Scale the fiducial C_L^phiphi by this factor (A_phi) "
                        "before anything is drawn from it. >1 is the "
                        "amplified-lensing validation (ROADMAP 2026-09-23): at "
                        "lmax<=300 a physical sky carries almost no lensing "
                        "information in T, so the sampler is stress-tested on a "
                        "sky with lensing strong enough to measure. The prior "
                        "the sampler uses is centred on the SAME amplified "
                        "spectrum, so the SBC remains a test of its own target.")
    p.add_argument("--blind", action="store_true",
                   help="Fit the lensing-BLIND model (Blocks 1-2 only, no phi) "
                        "to this realization's identical data map -- the "
                        "Commander-style baseline for the bias-reduction figure. "
                        "Same seed streams, so the sky matches the aware chain.")
    args = p.parse_args()
    if args.phi_amplitude <= 0:
        raise SystemExit("--phi_amplitude must be > 0")
    sample_cl_phiphi = not args.no_sample_cl_phiphi

    lmax, nside, r = args.lmax, args.nside, args.realization
    os.makedirs(args.outdir, exist_ok=True)
    tag = "blind" if args.blind else "chain"
    ckpt = os.path.join(args.outdir, f"{tag}_r{r:03d}_ckpt.npz")
    out = os.path.join(args.outdir, f"{tag}_r{r:03d}.npz")

    print(f"=== Coverage ensemble, realization {r} (lmax={lmax}, nside={nside}) ===")
    if args.cl_phiphi_prior_nu is not None:
        print(f"C_L^phiphi carries a PROPER conjugate prior (nu="
              f"{args.cl_phiphi_prior_nu}); the truth is drawn from that same "
              "joint prior, so the\nphi rank IS a valid SBC test here. C_l^TT is "
              "still flat/improper, so its row stays\ninterval coverage only.\n")
    else:
        print("Spectra held at fiducial; alm_true/phi_true drawn from their Gaussian "
              "priors.\nSpectrum-level ranks are interval coverage, NOT strict SBC "
              "(improper flat\nimplied prior -- see this script's header).\n")

    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=args.noisesig,
        data_mode="synthetic", dtype=tf.complex128, use_matrixfree_sht=True,
        lensing_operator=args.lensing_operator,
    )
    model._ensure_tf_tensors()
    assert len(model.unmasked_idx) == model.NPIX, "ensemble assumes full-sky data"

    if args.fiducial == "corrected":
        cl_true, cl_phiphi_fid = fiducial_spectra(lmax)
    else:
        cl_true = call_CAMB_map(LCDM_PARAMS, lmax)
        cl_phiphi_fid = get_cl_phiphi(lmax)
    cl_phiphi_fid = args.phi_amplitude * cl_phiphi_fid

    # With a PROPER prior on C_L^phiphi the generative process must match the
    # sampler's target, or the rank test is invalid in the other direction --
    # the same mistake, mirrored, that makes the Block-4-ON phi rank
    # uninterpretable today (truth from a proper N(0,C_fid), target improper).
    # So draw C_L^phiphi from the prior first, then phi from N(0, C_L).
    # The prior stays CENTRED ON THE FIDUCIAL (cl_phiphi_fid) -- that is what
    # run_gibbs_chain is told to use -- while the truth is one draw from it.
    if args.cl_phiphi_prior_nu is not None:
        nu = float(args.cl_phiphi_prior_nu)
        if nu <= 2.0:
            raise ValueError(f"--cl_phiphi_prior_nu must be > 2 (got {nu}); "
                             "the phi marginal is improper at or below 2.")
        rng_prior = np.random.default_rng(_STREAM_CLPP_PRIOR + r)
        cl_phiphi_true = np.zeros(lmax, dtype=np.float64)
        for ell in range(2, lmax):
            alpha0, beta0 = nu / 2.0, nu * cl_phiphi_fid[ell] / 2.0
            cl_phiphi_true[ell] = beta0 / rng_prior.gamma(alpha0, scale=1.0)
        print(f"C_L^phiphi truth drawn from the proper prior (nu={nu}); "
              f"draw/fiducial ratio min={np.min(cl_phiphi_true[2:lmax] / cl_phiphi_fid[2:lmax]):.3f} "
              f"max={np.max(cl_phiphi_true[2:lmax] / cl_phiphi_fid[2:lmax]):.3f}")
    else:
        cl_phiphi_true = cl_phiphi_fid

    alm_true_hp, phi_true_hp = _synalm_pair(
        cl_true, cl_phiphi_true, lmax, _STREAM_TRUTH + r
    )
    alm_true_packed = _alm_hp_to_packed(alm_true_hp, lmax)
    phi_true_packed = _alm_hp_to_packed(phi_true_hp, lmax)

    T_lensed_true = lens_map_tf(
        model, tf.constant(alm_true_packed, tf.float64), phi_true_hp
    ).numpy()
    if not np.all(np.isfinite(T_lensed_true)):
        raise RuntimeError(f"realization {r}: lens_map_tf produced NaN/Inf")

    rng_noise = np.random.default_rng(_STREAM_NOISE + r)
    noisy_map = T_lensed_true + rng_noise.normal(0.0, args.noisesig, size=model.NPIX)
    model.prior_map = noisy_map
    model.prior_map_masked = tf.convert_to_tensor(
        noisy_map[model.unmasked_idx], dtype=tf.float64
    )

    # Warm start from an independent prior draw, not a perturbation of truth.
    alm_start_hp, phi_start_hp = _synalm_pair(
        cl_true, cl_phiphi_true, lmax, _STREAM_START + r
    )
    alm_start_packed = _alm_hp_to_packed(alm_start_hp, lmax)
    phi_start_packed = _alm_hp_to_packed(phi_start_hp, lmax)
    start_corr = float(
        np.dot(phi_start_packed, phi_true_packed)
        / (np.linalg.norm(phi_start_packed) * np.linalg.norm(phi_true_packed))
    )
    print(f"  start-vs-truth phi cosine similarity={start_corr:+.4f} (~0 expected)")

    # --- alm start: the data-driven MAP, not the cold prior draw ---
    # MUST come after the data is installed on the model above, since psi (and
    # hence the MAP) depends on it. find_map_estimate ignores alm_start_packed
    # and descends from model.prior_parameters_tf() using the data alone, so
    # the truth never enters.
    #
    # !! This mirrors pilot_coverage_equilibration.py deliberately, and is NOT
    # optional. Omitting it is what broke the first coverage ensemble (job
    # 11887897, 2026-08-28): a cold prior-draw alm leaves a residual the phi
    # block absorbs by inflating phi ~30x in amplitude, and with Block 4 on the
    # refitted C_L^phiphi makes the phi prior EXACTLY scale-free (constant
    # sum_L (L-1.5); see test_block4_refitted_prior_is_scale_free_in_phi_amplitude)
    # so nothing pulls it back. Every realization froze 1e3-1e5x above the true
    # phi power and ranked 0/8 in every ell-bin. The same failure had already
    # been seen once, in job 11663105 (2026-07-30), which is why the pilot
    # carries the MAP start -- the ensemble script simply never inherited it,
    # so the equilibration gate and the production script were not running the
    # same pipeline.
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
        # the MAP is data-driven and the data contain the truth. Not a leak --
        # reported so "informed by data" vs "initialised at truth" stays
        # auditable, and because a near-zero value here predicts the failure.
        print(f"  MAP-vs-truth alm cosine similarity={map_corr:+.4f} "
              f"(expected >0: data-driven, not truth-initialised)")
    else:
        print("  ! map_steps=0: cold prior-draw alm start -- this is the "
              "configuration that broke jobs 11663105 and 11887897")
        x0 = np.concatenate([np.log(cl_true[2:lmax]), alm_start_packed])

    if args.blind:
        t0 = time.time()
        # No cl_phiphi_full -> no phi block: the Commander-style lensing-blind fit.
        samples, logp, accepts, _ = run_gibbs_chain(
            model, n_samples=args.n_samples, n_burnin=args.n_burnin,
            hmc_step_size=args.hmc_step_size, n_lfs=args.n_lfs,
            initial_params=x0, seed=r, checkpoint_path=ckpt,
            checkpoint_every=args.checkpoint_every)[:4]
        elapsed = time.time() - t0
        if not np.all(np.isfinite(samples)):
            raise RuntimeError(f"realization {r}: blind chain produced NaN/Inf")
        np.savez(out, packing_version=PACKING_VERSION, realization=r, lmax=lmax,
                 nside=nside, blind=True, alm_samples=samples, logp=logp,
                 accepts=accepts, cl_true=cl_true, cl_phiphi_true=cl_phiphi_true,
                 cl_phiphi_fid=cl_phiphi_fid, alm_true_packed=alm_true_packed,
                 phi_true_packed=phi_true_packed, n_burnin=args.n_burnin,
                 seconds_total=elapsed, lensing_operator=args.lensing_operator,
                 fiducial=args.fiducial, noisesig=args.noisesig,
                 phi_amplitude=args.phi_amplitude)
        print(f"  blind chain done in {elapsed / 3600:.2f}h; alm accept="
              f"{accepts.mean():.3f}; saved {out}")
        if os.path.exists(ckpt):
            os.remove(ckpt)
        return

    t0 = time.time()
    # run_gibbs_chain's return arity depends on which blocks are enabled: a
    # 5-tuple with Block 3 only, a 6-tuple with Block 3 + Block 4. Unpack
    # positionally rather than by fixed arity -- a hardcoded unpack here is
    # exactly the bug that crashed a completed 3000-sweep chain (achievements.md,
    # "Real bugs"), and SLURM still reported COMPLETED 0:0 because the traceback
    # landed after the job's own work was done.
    result = run_gibbs_chain(
        model,
        n_samples=args.n_samples,
        n_burnin=args.n_burnin,
        hmc_step_size=args.hmc_step_size,
        n_lfs=args.n_lfs,
        initial_params=x0,
        cl_phiphi_full=cl_phiphi_fid,
        phi_initial=phi_start_packed,
        phi_hmc_step_size=args.phi_hmc_step_size,
        phi_n_lfs=args.phi_n_lfs,
        phi_mass_matrix=args.phi_mass_matrix,
        phi_block_n_probes=args.phi_block_n_probes,
        sample_cl_phiphi=sample_cl_phiphi,
        cl_phiphi_prior_nu=args.cl_phiphi_prior_nu,
        seed=r,
        checkpoint_path=ckpt,
        checkpoint_every=args.checkpoint_every,
    )
    expected_arity = 6 if sample_cl_phiphi else 5
    if len(result) != expected_arity:
        raise RuntimeError(
            f"realization {r}: run_gibbs_chain returned {len(result)} values, "
            f"expected {expected_arity} for sample_cl_phiphi={sample_cl_phiphi}"
        )
    samples, phi_samples, logp, accepts, final_step = result[:5]
    cl_phiphi_samples = result[5] if sample_cl_phiphi else None
    elapsed = time.time() - t0
    print(f"  done in {elapsed / 3600:.2f}h; alm accept={accepts.mean():.3f}")

    finite_checks = [samples, phi_samples]
    if cl_phiphi_samples is not None:
        finite_checks.append(cl_phiphi_samples)
    if not all(np.all(np.isfinite(a)) for a in finite_checks):
        raise RuntimeError(f"realization {r}: NaN/Inf in samples")

    # --- calibration sanity check: computed here, RAISED AFTER THE SAVE ---
    # Job 11887897 wrote 12 finite, well-formed, completely wrong chains: phi
    # frozen 1.9e3-2.5e5x above the true power, and nothing downstream noticed
    # until the aggregate rank test ran. Finiteness is not calibration. This
    # compares the chain's phi power against the truth it was generated from
    # and fails LOUDLY rather than saving a plausible-looking corpse.
    #
    # The threshold is deliberately loose (100x). A correct-but-slowly-mixing
    # chain lands within a factor of a few (measured: 0.09-2.5 across every
    # lmax=64 pilot); the failure mode this guards against is 1e3-1e5. Anything
    # in between is genuinely ambiguous and worth a human look, which is what
    # the error message asks for.
    phi_power_chain = float(np.mean(phi_samples[len(phi_samples) // 2:] ** 2))
    phi_power_truth = float(np.mean(phi_true_packed ** 2))
    ratio = phi_power_chain / max(phi_power_truth, 1e-300)
    print(f"  phi power vs truth: chain={phi_power_chain:.4e} "
          f"truth={phi_power_truth:.4e} ratio={ratio:.3e}")
    phi_calibration_ok = 1e-2 < ratio < 1e2

    save_kwargs = {
        # Stamp the packing so a harvest can tell which model a chain describes:
        # k_L = 2L before the 2026-09-06 Im(a_{L,1}) restoration, 2L+1 after.
        # Chains with no field are version 1.
        "packing_version": PACKING_VERSION,
        "realization": r, "lmax": lmax, "nside": nside,
        "alm_samples": samples, "phi_samples": phi_samples,
        "logp": logp, "accepts": accepts,
        "cl_true": cl_true, "cl_phiphi_true": cl_phiphi_true,
        "cl_phiphi_fid": cl_phiphi_fid,
        "cl_phiphi_prior_nu": (np.nan if args.cl_phiphi_prior_nu is None
                               else float(args.cl_phiphi_prior_nu)),
        "alm_true_packed": alm_true_packed,
        "phi_true_packed": phi_true_packed,
        "start_cosine_similarity": start_corr,
        "n_burnin": args.n_burnin, "phi_n_lfs": args.phi_n_lfs,
        "phi_mass_matrix": args.phi_mass_matrix,
        "sample_cl_phiphi": sample_cl_phiphi,
        "seconds_total": elapsed,
        "seconds_per_sweep": elapsed / max(1, len(samples)),
        "map_steps": args.map_steps,
        "phi_power_ratio_to_truth": ratio,
        "phi_calibration_ok": phi_calibration_ok,
        "lensing_operator": args.lensing_operator, "fiducial": args.fiducial,
        "noisesig": args.noisesig, "phi_amplitude": args.phi_amplitude,
    }
    # Omit the key entirely (rather than storing None) when Block 4 is off, so
    # the aggregator's `"cl_phiphi_samples" in npz.files` check stays truthful
    # and no allow_pickle-requiring object array reaches the output.
    if cl_phiphi_samples is not None:
        save_kwargs["cl_phiphi_samples"] = cl_phiphi_samples
    np.savez(out, **save_kwargs)
    print(f"Saved realization {r} to {out}")

    # Raised only AFTER the save: the chain is hours of compute and is the
    # primary evidence for diagnosing whatever went wrong, so it is preserved
    # and the flag stored alongside it. The raise is what makes SLURM report
    # FAILED, so a broken configuration cannot quietly produce a full
    # directory of plausible-looking output the way job 11887897 did.
    if not phi_calibration_ok:
        raise RuntimeError(
            f"realization {r}: phi power is {ratio:.3e}x the truth (expected "
            f"O(1); every healthy lmax=64 pilot lands in 0.09-2.5). This is "
            f"the job-11887897 failure signature. First check the "
            f"MAP-vs-truth alm cosine similarity printed above: if it is near "
            f"zero, the alm block cold-started and phi inflated to absorb the "
            f"residual -- which Block 4's scale-free prior cannot correct "
            f"(ROADMAP.md item 7, achievements.md). Chain WAS saved to {out} "
            f"for diagnosis; do not aggregate it."
        )


if __name__ == "__main__":
    main()
