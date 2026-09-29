"""ROADMAP T0.1 (2026-09-27): exact dense reference for the a_lm power rank.

The a_lm `[lmax-4, lmax)` offset lives at l = lmax-1. The null it is read
against (`null_alm_power_rank_flat_prior.py`) is a DIAGONAL Gaussian model:
it ignores the pixelisation (A^T A is not proportional to the identity near
l ~ nside) and the lensing couplings. This script removes both approximations
for one piece of the problem: with phi FIXED at the sky's truth, the lensed
pixel likelihood is linear-Gaussian in the a_lm,

    d = A(phi_true) a + n,   n ~ N(0, sigma^2 I),

so the (C_l, a_lm) posterior under Block 1's flat C_l prior can be sampled
EXACTLY by a two-block Gibbs sampler with no Markov-chain error in either
block:

    a | C, d ~ N(P^-1 A^T d / sigma^2, P^-1),  P = A^T A / sigma^2 + diag(1/(C_l v_j)),
    C_l | a  ~ InvGamma(k_l/2 - 1, S_l/2)      (Block 1's own conditional).

A is built column by column from `lens_map_tf`, the production forward model,
and the data are rebuilt bit-for-bit from the production seeds, so each sky
here is the same sky as the production chain, minus the phi uncertainty.

If an exact sampler of THIS model also ranks the truth high at l = lmax-1,
the offset is a property of the pixelised model and the statistic (and the
diagonal null is what is wrong). If it does not, the production sampler is
the suspect.

Usage (one sky per array task):
  PYTHONPATH=diffcmb .venv/bin/python scripts/exact_dense_alm_reference.py \
      --chain results/analysis/ens_exact_l64_A3000_n30_nu30_long/chain_r000.npz \
      --outdir results/analysis/exact_dense_alm_reference_l64
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from null_alm_power_rank_flat_prior import packed_layout  # noqa: E402

from diffcmb.alm_utils import invgamma_shape_for_spectrum  # noqa: E402

STREAM_NOISE = 3_000_000  # = coverage_ensemble_chain._STREAM_NOISE (checked in main)


def build_operator(model, phi_hp, n_alm):
    """Dense A (n_pix, n_alm): column j = lens_map_tf(unit vector j)."""
    import tensorflow as tf

    from diffcmb.lensing import lens_map_tf

    cols = []
    e = np.zeros(n_alm)
    for j in range(n_alm):
        e[:] = 0.0
        e[j] = 1.0
        cols.append(lens_map_tf(model, tf.constant(e, tf.float64), phi_hp).numpy())
    return np.stack(cols, axis=1)


def draw_alm_given_cl(AtA, Atd, sigma2, C, L_arr, v, rng):
    """One exact draw of a | C, d ~ N(P^-1 A^T d / sigma^2, P^-1)."""
    from scipy.linalg import cho_factor, cho_solve, solve_triangular

    P = AtA / sigma2
    P[np.diag_indices_from(P)] += 1.0 / (C[L_arr] * v)
    c, low = cho_factor(P, lower=True, check_finite=False)
    mean = cho_solve((c, low), Atd / sigma2, check_finite=False)
    # x = mean + L^-T z has covariance (L L^T)^-1 = P^-1
    return mean + solve_triangular(c, rng.standard_normal(mean.size), lower=True,
                                   trans="T", check_finite=False)


def exact_gibbs(AtA, Atd, sigma2, lmax, n_sweeps, burn, thin, rng, cl_start):
    """(a draws, C draws) from the exact two-block Gibbs sampler."""
    from scipy.stats import invgamma

    L_arr, v = packed_layout(lmax)
    shape = invgamma_shape_for_spectrum(lmax, a0=-1.0)
    C = np.array(cl_start, dtype=np.float64)
    keep_a, keep_c = [], []
    for it in range(n_sweeps):
        a = draw_alm_given_cl(AtA, Atd, sigma2, C, L_arr, v, rng)
        S = np.bincount(L_arr, weights=a ** 2 / v, minlength=lmax)
        for ell in range(2, lmax):
            C[ell] = invgamma.rvs(shape[ell], scale=S[ell] / 2.0, random_state=rng)
        if it >= burn and (it - burn) % thin == 0:
            keep_a.append(a.copy())
            keep_c.append(C[2:].copy())
    return np.asarray(keep_a), np.asarray(keep_c)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain", required=True, help="production chain whose sky to rebuild")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_sweeps", type=int, default=1100)
    ap.add_argument("--burn", type=int, default=100)
    ap.add_argument("--thin", type=int, default=5)
    args = ap.parse_args()

    import coverage_ensemble_chain as cec
    import tensorflow as tf

    from diffcmb import CosmologyAdvancedSampling
    from diffcmb.lensing import _alm_packed_to_hp, lens_map_tf

    if cec._STREAM_NOISE != STREAM_NOISE:
        raise SystemExit("noise stream constant drifted from coverage_ensemble_chain")
    d0 = np.load(args.chain, allow_pickle=True)
    lmax, nside, r = int(d0["lmax"]), int(d0["nside"]), int(d0["realization"])
    sigma = float(d0["noisesig"])
    if str(d0["lensing_operator"]) != "exact":
        raise SystemExit("reference is built for the exact operator only")
    alm_true = np.asarray(d0["alm_true_packed"], dtype=np.float64)
    phi_hp = _alm_packed_to_hp(np.asarray(d0["phi_true_packed"], dtype=np.float64), lmax)
    cl_true = np.asarray(d0["cl_true"], dtype=np.float64)[:lmax]

    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=sigma, data_mode="synthetic",
        dtype=tf.complex128, use_matrixfree_sht=True, lensing_operator="exact")
    model._ensure_tf_tensors()

    t0 = time.time()
    A = build_operator(model, phi_hp, alm_true.size)
    print(f"r{r:03d}: built A {A.shape} in {time.time() - t0:.0f}s")
    # the production data, rebuilt from the production seeds
    T_lensed = A @ alm_true
    direct = lens_map_tf(model, tf.constant(alm_true, tf.float64), phi_hp).numpy()
    check = float(np.max(np.abs(direct - T_lensed)) / np.std(T_lensed))
    print(f"  linearity check: max |A a - lens(a)| / rms = {check:.2e}")
    if not check < 1e-8:
        raise SystemExit("A does not reproduce the forward model")
    rng_noise = np.random.default_rng(STREAM_NOISE + r)
    d = T_lensed + rng_noise.normal(0.0, sigma, size=model.NPIX)

    AtA = A.T @ A
    Atd = A.T @ d
    del A
    L_arr, v = packed_layout(lmax)
    S_true = np.bincount(L_arr, weights=alm_true ** 2 / v, minlength=lmax)
    k = np.bincount(L_arr, minlength=lmax)
    cl_start = np.where(k > 0, S_true / np.maximum(k, 1), cl_true)
    rng = np.random.default_rng(20260927 + r)
    t0 = time.time()
    a_s, c_s = exact_gibbs(AtA, Atd, sigma ** 2, lmax, args.n_sweeps, args.burn,
                           args.thin, rng, np.where(cl_start > 0, cl_start, 1.0))
    print(f"  exact Gibbs: {args.n_sweeps} sweeps in {time.time() - t0:.0f}s, "
          f"kept {a_s.shape[0]}")
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"chain_r{r:03d}.npz")
    # the layout of a production chain, so diagnose_alm_band_edge reads it as is
    np.savez(out, alm_samples=np.concatenate([np.log(c_s), a_s], axis=1),
             alm_true_packed=alm_true, cl_true=d0["cl_true"], lmax=lmax, nside=nside,
             noisesig=sigma, realization=r, phi_fixed_at_truth=True,
             source_chain=os.path.abspath(args.chain), AtA_diag=np.diag(AtA))
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
