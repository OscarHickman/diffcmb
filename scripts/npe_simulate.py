"""Training simulator for the learned posterior (ROADMAP T1.1).

theta = (ln C_l^TT, ln C_L^phiphi) for l, L = 2..lmax-1, drawn from the strict
proper priors; x = (ln data power spectrum, ln QE C_L^phiphi) of the noisy
lensed map. The generative process is coverage_ensemble_chain's strict mode
(--cl_prior_nu with --cl_phiphi_prior_nu) bit for bit -- same helpers, same
seed streams -- so a learned posterior trained here targets the same joint
posterior as the strict exact ensemble (tests/test_npe_simulate.py pins it).

The QE filter is the one validated at the production configuration
(scripts/validate_qe_noise.py output: cl_tt_lensed and nl_qe); the CLI refuses a
QE file whose configuration differs from the simulation's.

Keep training realizations disjoint from the exact ensemble's skies (0-47):
use --first >= 100000. A realization must stay below 1_000_000, the width of
each seed-stream block.
"""

import argparse
import os
import time
from dataclasses import dataclass

import healpy as hp
import numpy as np
import tensorflow as tf
from coverage_ensemble_chain import (
    _STREAM_CLPP_PRIOR,
    _STREAM_CLTT_PRIOR,
    _STREAM_NOISE,
    _STREAM_TRUTH,
    _synalm_pair,
    draw_from_invgamma_prior,
)

from diffcmb import CosmologyAdvancedSampling, qe
from diffcmb.lensing import _alm_hp_to_packed, lens_map_tf
from diffcmb.power import fiducial_spectra

_STREAM_WIDTH = 1_000_000


@dataclass(frozen=True)
class SimContext:
    model: CosmologyAdvancedSampling
    lmax: int
    nside: int
    noisesig: float
    phi_amplitude: float
    nu_tt: float
    nu_pp: float
    cl_tt_fid: np.ndarray
    cl_pp_fid: np.ndarray
    cl_tt_lensed: np.ndarray
    cl_tot: np.ndarray
    qe_norm: np.ndarray


def make_context(lmax, nside, noisesig, phi_amplitude, nu_tt, nu_pp,
                 cl_tt_lensed, nl_qe) -> SimContext:
    """One lensing model and QE filter, reused for every simulation."""
    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=noisesig, data_mode="synthetic",
        dtype=tf.complex128, use_matrixfree_sht=True, lensing_operator="exact",
    )
    model._ensure_tf_tensors()
    cl_tt_fid, cl_pp_fid = fiducial_spectra(lmax)
    cl_tt_lensed = np.asarray(cl_tt_lensed, dtype=np.float64)[: lmax + 1]
    nl_qe = np.asarray(nl_qe, dtype=np.float64)[: lmax + 1]
    return SimContext(
        model=model, lmax=lmax, nside=nside, noisesig=float(noisesig),
        phi_amplitude=float(phi_amplitude), nu_tt=float(nu_tt), nu_pp=float(nu_pp),
        cl_tt_fid=cl_tt_fid, cl_pp_fid=phi_amplitude * cl_pp_fid,
        cl_tt_lensed=cl_tt_lensed,
        cl_tot=cl_tt_lensed + qe.white_noise_cl(noisesig, 12 * nside ** 2, lmax),
        qe_norm=np.where(np.isfinite(nl_qe), nl_qe, 0.0),
    )


def summaries(ctx: SimContext, data_map: np.ndarray) -> np.ndarray:
    """ln data power spectrum and ln normalised-QE power, each for l = 2..lmax-1."""
    lmax = ctx.lmax
    cl_data = hp.anafast(data_map, lmax=lmax - 1)[2:lmax]
    phihat = hp.almxfl(
        qe.qe_tt_reconstruct(data_map, ctx.cl_tt_lensed, ctx.cl_tot, lmax, ctx.nside),
        ctx.qe_norm)
    cl_qe = hp.alm2cl(phihat, lmax=lmax)[2:lmax]
    return np.log(np.concatenate([cl_data, cl_qe]))


def simulate(ctx: SimContext, r: int) -> dict:
    """Realization r of coverage_ensemble_chain's strict-mode sky, plus summaries."""
    if not 0 <= r < _STREAM_WIDTH:
        raise ValueError(f"realization {r} is outside the seed-stream block [0, {_STREAM_WIDTH})")
    lmax = ctx.lmax
    cl_true = draw_from_invgamma_prior(ctx.cl_tt_fid, ctx.nu_tt, lmax, _STREAM_CLTT_PRIOR + r)
    cl_pp_true = draw_from_invgamma_prior(ctx.cl_pp_fid, ctx.nu_pp, lmax, _STREAM_CLPP_PRIOR + r)
    alm_hp, phi_hp = _synalm_pair(cl_true, cl_pp_true, lmax, _STREAM_TRUTH + r)
    alm_packed = _alm_hp_to_packed(alm_hp, lmax)
    lensed = lens_map_tf(ctx.model, tf.constant(alm_packed, tf.float64), phi_hp).numpy()
    if not np.all(np.isfinite(lensed)):
        raise RuntimeError(f"realization {r}: lens_map_tf produced NaN/Inf")
    rng_noise = np.random.default_rng(_STREAM_NOISE + r)
    data_map = lensed + rng_noise.normal(0.0, ctx.noisesig, size=lensed.size)
    return {
        "theta": np.log(np.concatenate([cl_true[2:lmax], cl_pp_true[2:lmax]])),
        "x": summaries(ctx, data_map),
        "cl_true": cl_true, "cl_phiphi_true": cl_pp_true,
        "alm_true_packed": alm_packed, "phi_true_packed": _alm_hp_to_packed(phi_hp, lmax),
        "data_map": data_map,
    }


def _check_qe_file(d, args):
    want = {"lmax": args.lmax, "nside": args.nside, "noisesig": args.noisesig,
            "phi_amplitude": args.phi_amplitude, "lensing_operator": "exact",
            "fiducial": "corrected", "passed": True}
    bad = []
    for key, val in want.items():
        got = d[key].item() if d[key].shape == () else d[key]
        if (got != val) if isinstance(val, (str, bool)) else not np.isclose(float(got), float(val)):
            bad.append(f"{key}: QE file has {got!r}, simulation uses {val!r}")
    if bad:
        raise SystemExit("QE file does not match the simulation:\n  " + "\n  ".join(bad))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lmax", type=int, default=64)
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--noisesig", type=float, default=30.0)
    p.add_argument("--phi_amplitude", type=float, default=3000.0)
    p.add_argument("--nu_tt", type=float, default=30.0)
    p.add_argument("--nu_pp", type=float, default=30.0)
    p.add_argument("--qe_file", default="results/analysis/qe_noise_validation_exact_A3000_n30.npz")
    p.add_argument("--first", type=int, required=True, help="first realization index")
    p.add_argument("--n", type=int, required=True, help="number of simulations")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    q = np.load(args.qe_file)
    _check_qe_file(q, args)
    ctx = make_context(args.lmax, args.nside, args.noisesig, args.phi_amplitude,
                       args.nu_tt, args.nu_pp, q["cl_tt_lensed"], q["nl_qe"])
    reals = np.arange(args.first, args.first + args.n)
    theta, x = [], []
    t0 = time.time()
    for i, r in enumerate(reals):
        sim = simulate(ctx, int(r))
        theta.append(sim["theta"])
        x.append(sim["x"])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.n} sims, {(time.time() - t0) / (i + 1):.2f} s/sim", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, theta=np.array(theta), x=np.array(x), realization=reals,
             lmax=args.lmax, nside=args.nside, noisesig=args.noisesig,
             phi_amplitude=args.phi_amplitude, nu_tt=args.nu_tt, nu_pp=args.nu_pp,
             qe_file=args.qe_file, lensing_operator="exact", fiducial="corrected")
    print(f"Saved {args.n} simulations to {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
