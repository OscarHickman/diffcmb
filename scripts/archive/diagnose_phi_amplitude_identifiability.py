"""Is the phi AMPLITUDE identified at all once Block 4 is on?

Motivated by the 2026-08-28 coverage-ensemble harvest (job 11887897), where every
realization's phi and C_L^phiphi ranked 0/8 in every ell-bin, with phi sitting
~7e4x above the truth in power and frozen there.

THE ANALYTIC POINT THIS SCRIPT TESTS NUMERICALLY
------------------------------------------------
Block 4 draws C_L^phiphi | phi ~ InvGamma(L-0.5, S_L/2), whose mean is
S_L/(2(L-1.5)). Substituting that back into log_prob_phi_block's Gaussian prior
term gives

    0.5 * sum_L S_L / C_L  =  0.5 * sum_L S_L * 2(L-1.5)/S_L  =  sum_L (L-1.5),

i.e. EXACTLY CONSTANT in the overall amplitude of phi. Once C_L^phiphi is
allowed to track phi, the prior exerts no restoring force on the scale at all
(this is the flat/improper implied prior that coverage_ensemble_chain.py's and
aggregate_coverage_ranks.py's headers already warn about, seen from the phi
side). The overall amplitude is then pinned by the lensing likelihood ALONE.

So the question that decides whether the ensemble failure is a warm-start bug or
a fundamental identifiability problem is: at this (lmax, nside, noise), does the
lensing likelihood actually have a minimum at the true phi amplitude?

WHAT IT COMPUTES
----------------
Along the pure scaling ray phi(s) = s * phi_true, for a grid of s spanning the
observed failure (s=1 is truth, s~265 is where the ensemble chain froze), it
reports the three pieces of the phi-block target separately:

  neg_log_lik(s)      -- psi_lensed, the lensing likelihood only
  prior_fixed(s)      -- Gaussian prior at the FIXED fiducial C_L^phiphi (Block 4 OFF)
  prior_block4(s)     -- Gaussian prior at C_L^phiphi refitted to phi(s) (Block 4 ON)

and does it twice: once with the alm held at TRUTH, once with the alm held at an
independent prior draw (the ensemble's actual cold start). The second case tests
whether a wrong alm actively drives phi upward, which is the proposed trigger for
the runaway during the ensemble's 100-sweep burn-in.

Read the output as:
  * prior_block4 flat in s   -> confirms the analytic result above.
  * neg_log_lik minimised at s=1 with alm=truth -> amplitude IS identified by the
    likelihood; the ensemble failure is a warm-start/burn-in bug (fix: MAP).
  * neg_log_lik flat or minimised at s>>1 -> amplitude is NOT identified at this
    configuration; Block 4 is ill-posed here and no warm start rescues it.

Deliberately computes no verdict beyond printing the profiles -- the numbers are
the evidence.

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/archive/diagnose_phi_amplitude_identifiability.py \
      --lmax 64 --nside 64
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

from diffcmb import CosmologyAdvancedSampling
from diffcmb.lensing import (
    _alm_hp_to_packed,
    compute_sl_phi_np,
    lens_map_tf,
    log_prob_phi_block,
    psi_lensed,
)
from diffcmb.power import call_CAMB_map

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/
import _paths  # noqa: E402,F401  (puts every script folder on sys.path)
from coverage_ensemble_chain import (  # noqa: E402
    _STREAM_NOISE,
    _STREAM_START,
    _STREAM_TRUTH,
    LCDM_PARAMS,
    _synalm_pair,
    get_cl_phiphi,
)


def block4_cl_from_phi(phi_packed, lmax):
    """C_L^phiphi refitted to phi -- the mean of Block 4's InvGamma conditional.

    Uses the conditional MEAN S_L/(2(L-1.5)) rather than a random draw so the
    profile is deterministic. L=2 has alpha=1.5 so the mean is S_L itself; the
    (L-1.5) factor is kept general and floored to stay positive.
    """
    S = compute_sl_phi_np(phi_packed, lmax)
    cl = np.zeros(lmax, dtype=np.float64)
    for L in range(2, lmax):
        alpha_minus_1 = max(L - 1.5, 0.5)
        cl[L] = max(S[L] / (2.0 * alpha_minus_1), 1e-30)
    return cl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lmax", type=int, default=64)
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--noisesig", type=float, default=1.0)
    p.add_argument("--realization", type=int, default=0)
    args = p.parse_args()

    lmax, nside, r = args.lmax, args.nside, args.realization
    print(f"=== phi amplitude identifiability: lmax={lmax} nside={nside} "
          f"noisesig={args.noisesig} realization={r} ===\n")

    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=args.noisesig,
        data_mode="synthetic", dtype=tf.complex128, use_matrixfree_sht=True,
    )
    model._ensure_tf_tensors()

    cl_true = call_CAMB_map(LCDM_PARAMS, lmax)
    cl_pp_true = get_cl_phiphi(lmax)

    alm_true_hp, phi_true_hp = _synalm_pair(cl_true, cl_pp_true, lmax, _STREAM_TRUTH + r)
    alm_true_packed = _alm_hp_to_packed(alm_true_hp, lmax)
    phi_true_packed = _alm_hp_to_packed(phi_true_hp, lmax)

    # Data exactly as the ensemble builds it.
    T_lensed_true = lens_map_tf(
        model, tf.constant(alm_true_packed, tf.float64), phi_true_hp
    ).numpy()
    rng_noise = np.random.default_rng(_STREAM_NOISE + r)
    noisy_map = T_lensed_true + rng_noise.normal(0.0, args.noisesig, size=model.NPIX)
    model.prior_map = noisy_map
    model.prior_map_masked = tf.convert_to_tensor(
        noisy_map[model.unmasked_idx], dtype=tf.float64
    )

    # How big is the lensing signal relative to the noise at all? If the lensed
    # and unlensed maps differ by much less than the noise, no amount of
    # sampling can identify phi.
    T_unlensed = lens_map_tf(
        model, tf.constant(alm_true_packed, tf.float64),
        np.zeros_like(phi_true_hp)
    ).numpy()
    d_rms = float(np.std(T_lensed_true - T_unlensed))
    print(f"lensing signal rms |T_lensed - T_unlensed| = {d_rms:.4e}")
    print(f"noise rms                                  = {args.noisesig:.4e}")
    print(f"=> per-pixel lensing S/N                   = {d_rms/args.noisesig:.4e}")
    print(f"   whole-map lensing S/N (x sqrt(Npix))    = "
          f"{d_rms/args.noisesig*np.sqrt(model.NPIX):.4e}\n")

    alm_start_hp, _ = _synalm_pair(cl_true, cl_pp_true, lmax, _STREAM_START + r)
    alm_start_packed = _alm_hp_to_packed(alm_start_hp, lmax)

    lncl = np.log(cl_true[2:lmax])
    scales = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 265.0, 1000.0]

    for alm_label, alm_packed in (("alm = TRUTH", alm_true_packed),
                                  ("alm = COLD PRIOR DRAW (ensemble start)",
                                   alm_start_packed)):
        params = tf.constant(np.concatenate([lncl, alm_packed]), tf.float64)
        print(f"--- {alm_label} ---")
        print(f"{'s':>8} {'neg_log_lik':>16} {'prior_fixedC':>16} "
              f"{'prior_block4C':>16} {'target_b4(-)':>16}")
        rows = []
        for s in scales:
            phi = s * phi_true_packed
            phi_tf = tf.constant(phi, tf.float64)
            nll = float(psi_lensed(model, params, phi_tf).numpy())
            # Block 4 OFF: prior at the fiducial spectrum.
            lp_fixed = float(
                log_prob_phi_block(model, params, phi_tf, cl_pp_true).numpy()
            )
            prior_fixed = -lp_fixed - nll
            # Block 4 ON: prior at the spectrum refitted to this phi.
            cl_b4 = block4_cl_from_phi(phi, lmax)
            lp_b4 = float(log_prob_phi_block(model, params, phi_tf, cl_b4).numpy())
            prior_b4 = -lp_b4 - nll
            rows.append((s, nll, prior_fixed, prior_b4, nll + prior_b4))
            print(f"{s:8.1f} {nll:16.6e} {prior_fixed:16.6e} "
                  f"{prior_b4:16.6e} {nll + prior_b4:16.6e}")
        arr = np.array(rows)
        i_nll = int(np.argmin(arr[:, 1]))
        i_fix = int(np.argmin(arr[:, 2] + arr[:, 1]))
        i_b4 = int(np.argmin(arr[:, 4]))
        print(f"  argmin neg_log_lik            at s = {arr[i_nll,0]:g}")
        print(f"  argmin (lik + fixed-C prior)  at s = {arr[i_fix,0]:g}   "
              f"[Block 4 OFF target]")
        print(f"  argmin (lik + block4-C prior) at s = {arr[i_b4,0]:g}   "
              f"[Block 4 ON target]")
        spread = arr[:, 3].max() - arr[:, 3].min()
        print(f"  block4 prior spread over the whole ray = {spread:.6e} "
              f"(0 => scale-free, as predicted analytically)\n")


if __name__ == "__main__":
    main()
