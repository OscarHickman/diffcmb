"""Validate diffcmb/qe.py's N_L^phiphi against the QE's own definition, by Monte Carlo.

WHY THIS EXISTS. plots/STORY.md carries a standing instruction that paper
figure 3's quadratic-estimator noise curve must not be fabricated: it has to be
validated before it appears in a figure. This is that validation.
`scripts/paper/fig3_uncertainty_propagation.py` refuses to build without the
artifact this script writes, and achievements.md records the result.

A FAILED FIRST DESIGN, kept here because it is an easy trap to fall into twice.
The first version compared N_L against `lensing.estimate_phi_diag_fisher`, on
the reasoning that the QE Fisher is 1/N_L and the two routes are independent.
They are independent, but they are not the same quantity:
`estimate_phi_diag_fisher` is the curvature of psi_lensed at FIXED alm -- the
information about phi when the unlensed CMB is known exactly -- whereas N_L is
the reconstruction noise when the CMB is unknown and averaged over. Knowing the
true CMB is worth far more information than the QE has, and the comparison came
out at a median ratio of 132 with a +0.66 trend in ln L (job 12015538). That is
not a bug in either piece of code; it is a category error in the test.

THE TEST THAT REPLACED IT. N_L is defined by what the estimator does, so the
estimator is run. Two Monte Carlo checks, which must pass TOGETHER:

  (1) RESPONSE, on skies lensed by a known phi. The normalised estimator must
      return the input:  <phihat_L . phi_true_L> / <phi_true_L . phi_true_L> = 1.
  (2) NOISE, on unlensed skies (phi = 0). The normalised estimator's variance
      must be the predicted noise:  Var(phihat_LM) / N_L = 1.

Neither alone is sufficient, and that is the point. A scale error anywhere --
in A_L, in the position-space estimator, in a spin convention -- multiplies the
two tests by c and 1/c^2 respectively, so no single constant can make both pass
at once. Passing both therefore validates the normalisation A_L (which IS the
noise curve) rather than merely demonstrating internal consistency. NO FREE
NORMALISATION IS FITTED IN THIS SCRIPT.

A PASS is response ~ 1 and noise ratio ~ 1, both with no trend in L. A flat
offset in one only indicts the estimator's convention; a tilt in L indicts the
coupling f inside A_L; a matched offset in both indicts the filter spectra.

Usage:
  sbatch scripts/submit_validate_qe_noise.slurm
"""

import argparse
import os
import sys

import healpy as hp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "diffcmb"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffcmb import qe  # noqa: E402
from diffcmb.power import call_CAMB_map, fiducial_spectra  # noqa: E402

LCDM_PARAMS = [67.74, 0.0486, 0.2589, 0.06, 0.0, 0.066]

# L bins for reporting. Matched to the coverage ensemble's bins so the figure,
# the chains and this validation all speak about the same ranges.
L_BINS = [(2, 10), (10, 30), (30, 60), (60, 64)]


def get_cl_phiphi(lmax):
    from coverage_ensemble_chain import get_cl_phiphi as _g
    return _g(lmax)


def _cross_ratio(alm_a, alm_b, lmax, ells):
    """<a . b> / <b . b> over a set of multipoles -- the response estimator."""
    cab = hp.alm2cl(alm_a, alm_b, lmax=lmax)
    cbb = hp.alm2cl(alm_b, alm_b, lmax=lmax)
    num = np.sum((2 * ells + 1) * cab[ells])
    den = np.sum((2 * ells + 1) * cbb[ells])
    return num / den if den > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lmax", type=int, default=64)
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--noisesig", type=float, default=1.0)
    ap.add_argument("--n_sims", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/analysis/qe_noise_validation.npz")
    ap.add_argument("--fiducial", choices=("corrected", "legacy"), default="corrected")
    ap.add_argument("--phi_amplitude", type=float, default=1.0,
                    help="A_phi, must match the chains (coverage_ensemble_chain.py).")
    ap.add_argument("--lensing_operator", choices=("exact", "bilinear"), default="exact")
    args = ap.parse_args()

    lmax, nside = args.lmax, args.nside
    npix = 12 * nside ** 2
    print(f"lmax={lmax} nside={nside} noisesig={args.noisesig} "
          f"n_sims={args.n_sims}\n", flush=True)

    import tensorflow as tf

    from diffcmb.lensing import _alm_hp_to_packed, lens_map_tf
    from diffcmb.model import CosmologyAdvancedSampling

    if args.fiducial == "corrected":
        tt_unl, cl_pp = fiducial_spectra(lmax + 1)
        cl_pp = cl_pp[:lmax]
    else:
        tt_unl = np.zeros(lmax + 1)
        legacy = call_CAMB_map(LCDM_PARAMS, lmax)
        tt_unl[:len(legacy)] = legacy[:lmax + 1]
        cl_pp = get_cl_phiphi(lmax)
    cl_pp = args.phi_amplitude * cl_pp
    tt_unl = tt_unl[:lmax + 1]

    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=args.noisesig,
        data_mode="synthetic", dtype=tf.complex128, use_matrixfree_sht=True,
        lensing_operator=args.lensing_operator,
    )
    model._ensure_tf_tensors()

    # ---- lensed skies with known phi (used twice: to MEASURE the lensed TT
    # spectrum the estimator's weights and filter need, and for the response
    # test). With amplified lensing the lensed spectrum differs visibly from
    # the unlensed one, so it is measured from the same operator the chains
    # use rather than assumed.
    lensed = []
    for i in range(args.n_sims):
        np.random.seed(args.seed + 500000 + i)
        tlm = hp.synalm(tt_unl[:lmax], lmax=lmax - 1, new=True).astype(np.complex128)
        plm = hp.synalm(cl_pp, lmax=lmax - 1, new=True).astype(np.complex128)
        tmap = lens_map_tf(
            model, tf.constant(_alm_hp_to_packed(tlm, lmax), tf.float64), plm
        ).numpy()
        lensed.append((tmap, plm))
        if (i + 1) % 16 == 0:
            print(f"  lensed sim {i + 1}/{args.n_sims}", flush=True)
    cl_tt = np.mean([hp.anafast(m, lmax=lmax, iter=3) for m, _ in lensed], axis=0)
    cl_tt[:2] = 0.0

    nl_noise = qe.white_noise_cl(args.noisesig, npix, lmax)
    cl_tot = cl_tt + nl_noise
    nl_qe = qe.qe_tt_noise_nl(cl_tt, cl_tot, lmax)
    print(f"N_L computed from the measured lensed spectrum; finite for "
          f"L=2..{lmax - 1}\n", flush=True)

    ell = np.arange(lmax + 1)
    norm = np.where(np.isfinite(nl_qe), nl_qe, 0.0)  # A_L = N_L

    # ---- (2) NOISE: unlensed skies, phi = 0 ------------------------------
    rng = np.random.default_rng(args.seed)
    cl_hat_acc = np.zeros(lmax + 1)
    for i in range(args.n_sims):
        np.random.seed(args.seed + 900000 + i)
        tlm = hp.synalm(cl_tt, lmax=lmax, new=True)
        tmap = hp.alm2map(tlm, nside, lmax=lmax)
        tmap = tmap + rng.normal(0.0, args.noisesig, size=npix)
        phihat = hp.almxfl(
            qe.qe_tt_reconstruct(tmap, cl_tt, cl_tot, lmax, nside), norm)
        cl_hat_acc += hp.alm2cl(phihat, lmax=lmax)
        if (i + 1) % 16 == 0:
            print(f"  noise sim {i + 1}/{args.n_sims}", flush=True)
    cl_hat_noise = cl_hat_acc / args.n_sims

    # ---- (1) RESPONSE: the lensed skies above, with noise -------------------
    resp_num = np.zeros(lmax + 1)
    resp_den = np.zeros(lmax + 1)
    for tmap, plm in lensed:
        tmap = tmap + rng.normal(0.0, args.noisesig, size=npix)
        phihat = hp.almxfl(
            qe.qe_tt_reconstruct(tmap, cl_tt, cl_tot, lmax, nside), norm)
        plm_pad = hp.resize_alm(plm, lmax - 1, lmax - 1, lmax, lmax)
        resp_num += hp.alm2cl(phihat, plm_pad, lmax=lmax)
        resp_den += hp.alm2cl(plm_pad, plm_pad, lmax=lmax)

    # ---- report ----------------------------------------------------------
    print(f"\n{'L bin':>12} {'response':>10} {'noise ratio':>12}")
    rows = []
    for lo, hi in L_BINS:
        ells = np.arange(max(2, lo), min(lmax, hi))
        if ells.size == 0:
            continue
        w = 2 * ells + 1
        resp = (np.sum(w * resp_num[ells]) / np.sum(w * resp_den[ells])
                if np.sum(w * resp_den[ells]) > 0 else np.nan)
        nratio = (np.sum(w * cl_hat_noise[ells]) / np.sum(w * nl_qe[ells])
                  if np.all(np.isfinite(nl_qe[ells])) else np.nan)
        rows.append((lo, hi, resp, nratio))
        print(f"  [{lo:3d},{hi:3d})  {resp:10.3f} {nratio:12.3f}")

    resp_all = np.array([r[2] for r in rows])
    nr_all = np.array([r[3] for r in rows])
    print(f"\n  response   : median {np.nanmedian(resp_all):.3f}  "
          f"(1.000 = estimator returns the input phi)")
    print(f"  noise ratio: median {np.nanmedian(nr_all):.3f}  "
          f"(1.000 = N_L is the measured variance)")
    ok = (abs(np.nanmedian(resp_all) - 1) < 0.15
          and abs(np.nanmedian(nr_all) - 1) < 0.15)
    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'} "
          "-- both must sit at 1; see this script's docstring for what a "
          "one-sided offset vs a tilt in L each indict.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, L=ell, nl_qe=nl_qe, cl_hat_noise=cl_hat_noise,
             resp_num=resp_num, resp_den=resp_den,
             bins=np.array([(r[0], r[1]) for r in rows]),
             response=resp_all, noise_ratio=nr_all,
             median_ratio=float(np.nanmedian(nr_all)),
             median_response=float(np.nanmedian(resp_all)),
             passed=bool(ok), lmax=lmax, nside=nside,
             noisesig=args.noisesig, n_sims=args.n_sims, cl_tt_lensed=cl_tt,
             cl_pp=cl_pp, phi_amplitude=args.phi_amplitude,
             fiducial=args.fiducial, lensing_operator=args.lensing_operator)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
