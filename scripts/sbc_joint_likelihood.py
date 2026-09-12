"""SBC on the JOINT LIKELIHOOD as a test quantity (Modrak et al., arXiv:2211.02383).

WHY THIS EXISTS. Modrak, Moon, Kim, Burkner, Huurre, Faltejskova, Gelman &
Vehtari, "Simulation-Based Calibration Checking for Bayesian Computation: The
Choice of Test Quantities Shapes Sensitivity", show that SBC's power to detect a
broken sampler depends on WHICH test quantity is ranked, and that
*data-dependent* quantities -- the joint likelihood above all -- catch failures
that parameter-wise ranks are blind to (including posterior-equals-prior, which
no parameter rank can see). This project currently ranks only parameter-derived
quantities: per-l-bin field power and the spectra. This script adds the
recommended one.

THE TEST. For each realization the test quantity is the log-likelihood

    L(alm, phi) = -0.5 * sum_pix Ninv * (y - lens(alm, phi))^2

evaluated at every posterior sweep and at the TRUTH. Under a correct sampler,
rank(L_true among {L_i}) is Uniform -- no null needed, unlike the spectrum
coverage rows (dashboard.md). A LOW rank means the posterior draws fit the data
better than the truth does, i.e. the sampler is overfitting the noise; a HIGH
rank means they fit it worse.

WHY THE RECONSTRUCTION IS SAFE. The saved chains carry (alm_samples,
phi_samples, logp) but NOT the data map, so the likelihood cannot be evaluated
without rebuilding y. Rebuilding means replaying coverage_ensemble_chain.py's
RNG, and a gate that reimplements the production script's setup is exactly how
the cold-start bug reached a full 12-chain ensemble (achievements.md). So this
script does not trust its own reconstruction: it imports the stream constants
and the draw helper FROM coverage_ensemble_chain, then ASSERTS that the
reconstructed alm_true/phi_true match the `alm_true_packed`/`phi_true_packed`
saved in the chain file to within 1e-12 before using anything. A stream that has
drifted fails loudly instead of silently producing a plausible wrong rank.
Everything downstream of the truth (the lensed map, the noise draw) is a
deterministic function of those plus the noise seed.

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/sbc_joint_likelihood.py \
      --indir results/analysis/coverage_ensemble_lmax64_prior_nocl4_packingv2 \
      --noisesig 1.0 --thin 10
"""

import argparse
import glob
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "diffcmb"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf  # noqa: E402
from coverage_ensemble_chain import (  # noqa: E402
    _STREAM_NOISE,
    _STREAM_TRUTH,
    _synalm_pair,
)

from diffcmb import CosmologyAdvancedSampling  # noqa: E402
from diffcmb.lensing import (  # noqa: E402
    _alm_hp_to_packed,
    _alm_packed_to_hp,
    lens_map_tf,
)

TOL = 1e-12


def rebuild_truth_and_data(model, d, lmax, noisesig):
    """Replay the chain's generative path; verify against what it saved.

    The spectra are read from the CHAIN FILE rather than recomputed from CAMB.
    A first version called call_CAMB_map/get_cl_phiphi here and tripped the
    verification below at 2.3e-7 on realization 1 -- far too large for a
    float64 replay of the same draw, far too small for a wrong RNG stream. The
    cause is CAMB: it is not bit-reproducible across runs/threadings, so a
    recomputed C_l differs in the last few digits, and hp.synalm amplifies that
    into the truth alm. The saved `cl_true`/`cl_phiphi_true` are the exact
    arrays the chain drew from, so using them removes CAMB from the replay
    entirely and leaves only numpy's deterministic stream.
    """
    r = int(d["realization"])
    cl_true = np.asarray(d["cl_true"], dtype=np.float64)
    cl_phiphi_true = np.asarray(d["cl_phiphi_true"], dtype=np.float64)

    alm_true_hp, phi_true_hp = _synalm_pair(
        cl_true, cl_phiphi_true, lmax, _STREAM_TRUTH + r
    )
    alm_true_packed = _alm_hp_to_packed(alm_true_hp, lmax)
    phi_true_packed = _alm_hp_to_packed(phi_true_hp, lmax)

    # The load-bearing check: if the streams drifted, stop.
    for nm, got, want in (("alm", alm_true_packed, d["alm_true_packed"]),
                          ("phi", phi_true_packed, d["phi_true_packed"])):
        err = float(np.max(np.abs(got - want)))
        if err > TOL:
            raise SystemExit(
                f"realization {r}: reconstructed {nm}_true disagrees with the "
                f"value saved in the chain (max abs diff {err:.3e} > {TOL:g}). "
                f"The RNG replay has drifted from coverage_ensemble_chain.py -- "
                f"refusing to compute a likelihood against a wrong data map."
            )

    T_lensed = lens_map_tf(
        model, tf.constant(alm_true_packed, tf.float64), phi_true_hp
    ).numpy()
    rng_noise = np.random.default_rng(_STREAM_NOISE + r)
    y = T_lensed + rng_noise.normal(0.0, noisesig, size=model.NPIX)
    return alm_true_packed, phi_true_packed, y


def loglik(model, y, alm_packed, phi_packed, lmax):
    phi_hp = _alm_packed_to_hp(np.asarray(phi_packed, dtype=np.float64), lmax)
    T = lens_map_tf(model, tf.constant(np.asarray(alm_packed, np.float64),
                                       tf.float64), phi_hp).numpy()
    resid = y - T
    return float(-0.5 * np.sum(model.Ninv * resid * resid))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indir", required=True)
    ap.add_argument("--noisesig", type=float, default=1.0)
    ap.add_argument("--thin", type=int, default=10,
                    help="evaluate every Nth sweep (each costs one lensing op)")
    ap.add_argument("--burn_frac", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0,
                    help="only process the first N realizations (smoke test)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "chain_r[0-9][0-9][0-9].npz")))
    if not files:
        raise SystemExit(f"no chain files in {args.indir}")
    if args.limit:
        files = files[:args.limit]
    print(f"SBC on the joint likelihood -- {len(files)} realizations, "
          f"thin={args.thin}\n")

    model = None
    ranks, n_draws = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        lmax, nside = int(d["lmax"]), int(d["nside"])
        if model is None:
            model = CosmologyAdvancedSampling(
                _lmax=lmax, _NSIDE=nside, _noisesig=args.noisesig,
                data_mode="synthetic", dtype=tf.complex128,
                use_matrixfree_sht=True,
            )
            model._ensure_tf_tensors()

        alm_t, phi_t, y = rebuild_truth_and_data(model, d, lmax, args.noisesig)
        model.prior_map = y
        model.prior_map_masked = tf.convert_to_tensor(
            y[model.unmasked_idx], dtype=tf.float64
        )

        alm_s, phi_s = d["alm_samples"], d["phi_samples"]
        n = alm_s.shape[0]
        lo = int(args.burn_frac * n)
        idx = np.arange(lo, n, args.thin)
        n_cl = lmax - 2
        L = np.array([loglik(model, y, alm_s[i, n_cl:], phi_s[i], lmax)
                      for i in idx])
        L_true = loglik(model, y, alm_t, phi_t, lmax)

        # Rank of the truth among the draws, mapped to (0,1).
        u = float((np.sum(L < L_true) + 0.5 * np.sum(L == L_true)) / len(L))
        ranks.append(u)
        n_draws.append(len(L))
        print(f"  {os.path.basename(f)}: u={u:.4f}  "
              f"(L_true={L_true:.4e}, draws {len(L)})")

    ranks = np.array(ranks)
    ks = stats.kstest(ranks, "uniform")
    print(f"\n  N={len(ranks)}  mean_u={ranks.mean():.4f}  KS_p={ks.pvalue:.4g}")
    print(f"  uniform expects 0.500 +/- {0.2887/np.sqrt(len(ranks)):.4f}")
    if ks.pvalue < 0.01:
        print("  -> REJECTED. The joint likelihood is a data-dependent test "
              "quantity, so this is sensitive to failures the field-power "
              "ranks cannot see (Modrak et al.). mean_u < 0.5 means the "
              "posterior fits the data BETTER than the truth does (overfitting "
              "the noise); > 0.5 means worse.")
    else:
        print("  -> consistent with uniform.")

    out = args.out or os.path.join(args.indir, "sbc_joint_likelihood.npz")
    np.savez(out, ranks=ranks, n_draws=np.array(n_draws),
             mean_u=ranks.mean(), ks_p=ks.pvalue, thin=args.thin)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
