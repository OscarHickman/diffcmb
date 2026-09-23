"""How much of the lensing-blind C_l^TT deficit is physics, and how much is operator?

Paper figure 2 reports that a lensing-blind fit to the simulated data leaves a
C_l^TT deficit that deepens with l (to -5.4% at [100,128) for lmax=128), and
that the joint sampler removes it. The aware sampler is unbiased because its
forward model contains the *same* operator as the simulation, so that part is
sound. What figure 2 does not say is what the deficit is made of. Three things
can remove in-band power from the simulated lensed map:

  1. physical lensing (smoothing of the acoustic structure) -- the only part
     that exists on the real sky;
  2. band-limit truncation: the unlensed simulation has no power at l >= lmax,
     so lensing only moves power *out* of the band, never in;
  3. the operator's numerics: diffcmb lenses by HEALPix bilinear interpolation
     (hp.get_interp_weights) on an nside = lmax grid, which smooths.

This script separates them by lensing the same Gaussian realizations five ways
and measuring <|a_lm^lensed|^2> / <|a_lm^unlensed|^2> per l (ratio of sums over
realizations):

  A  diffcmb operator: precompute_lensing bilinear weights, nside = lmax,
     analysed at nside = lmax  (what the production chains see)
  B  exact evaluation (ducc0 synthesis_general) at the SAME deflected angles,
     nside = lmax  -> A vs B isolates the interpolation error
  Bh as B, on an nside_hi grid -> B vs Bh isolates aliasing at nside = lmax
  C  as Bh, but the unlensed input extended to 3*lmax -> Bh vs C isolates
     band-limit truncation; C is what a real (unbandlimited) sky would do
  D  CAMB lensed/unlensed TT ratio (full-sky theory, all L of phi)

The same deflected angles (theta + d_theta, phi + d_phi, as precompute_lensing
builds them) are used by A, B, Bh and C, so the comparison is of interpolation
and band limits, not of the displacement convention.

Usage (SLURM, see submit_validate_lensing_power_transfer.slurm):
  PYTHONPATH=diffcmb python scripts/validate_lensing_power_transfer.py \
      --lmax_list 64 128 192 --n_real 16 --out results/analysis/lensing_power_transfer.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import healpy as hp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffcmb.lensing import deflection_field, precompute_lensing  # noqa: E402

BINS = [(2, 10), (10, 30), (30, 60), (60, 100), (100, 128), (128, 160),
        (160, 192)]
# Same fiducial as every production script (coverage_ensemble_chain.LCDM_PARAMS).
LCDM = [67.74, 0.0486, 0.2589, 0.06, 0.0, 0.066]


def camb_spectra(lmax_out: int):
    """Unlensed TT, lensed TT (muK^2, C_l) and C_L^phiphi, indexed l=0..lmax_out-1."""
    import camb

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=LCDM[0], ombh2=LCDM[1], omch2=LCDM[2], mnu=LCDM[3],
                       omk=LCDM[4], tau=LCDM[5])
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(lmax_out + 500, lens_potential_accuracy=1)
    res = camb.get_results(pars)
    pw = res.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)
    unl = pw["unlensed_scalar"][:lmax_out, 0]
    len_ = pw["lensed_scalar"][:lmax_out, 0]
    pp = res.get_lens_potential_cls(lmax=lmax_out - 1, raw_cl=True)[:lmax_out, 0]
    return unl, len_, pp


def exact_eval(alm, lmax_hp, theta, phi, nthreads):
    import ducc0

    loc = np.ascontiguousarray(np.stack([theta, phi % (2 * np.pi)], axis=1))
    out = ducc0.sht.synthesis_general(
        alm=np.ascontiguousarray(alm[None, :].astype(np.complex128)),
        spin=0, lmax=lmax_hp, loc=loc, epsilon=1e-10, nthreads=nthreads)
    return out[0]


def deflected_angles(phi_alm, nside, lmax):
    pix = np.arange(hp.nside2npix(nside))
    d_th, d_ph = deflection_field(phi_alm, nside, lmax)
    th0, ph0 = hp.pix2ang(nside, pix)
    return np.clip(th0 + d_th, 1e-12, np.pi - 1e-12), ph0 + d_ph


def power(m, lmax_hp):
    return hp.anafast(m, lmax=lmax_hp, iter=3)


def run_lmax(lmax, n_real, cl_unl_ext, cl_pp, nside_hi_fac, nthreads, seed0):
    lh = lmax - 1                    # healpy lmax of the band-limited field
    lh_ext = 3 * lmax - 1
    nside = lmax
    nside_hi = nside_hi_fac * lmax
    sums = {k: np.zeros(lmax) for k in ("unl", "A", "B", "Bh", "C")}
    rng = np.random.default_rng(seed0 + lmax)
    for r in range(n_real):
        np.random.seed(int(rng.integers(2**31)))
        alm_ext = hp.synalm(cl_unl_ext[: lh_ext + 1], lmax=lh_ext, new=True)
        # In-band part: identical modes, truncated at l = lmax-1.
        alm_T = np.zeros(hp.Alm.getsize(lh), dtype=complex)
        for m in range(lh + 1):
            for ell in range(m, lh + 1):
                alm_T[hp.Alm.getidx(lh, ell, m)] = alm_ext[hp.Alm.getidx(lh_ext, ell, m)]
        cl_pp_band = cl_pp[: lh + 1].copy()
        cl_pp_band[:2] = 0.0
        phi_alm = hp.synalm(cl_pp_band, lmax=lh, new=True)

        sums["unl"] += hp.alm2cl(alm_T)

        # A: diffcmb operator (bilinear on the nside = lmax grid).
        T_map = hp.alm2map(alm_T, nside, lmax=lh)
        nb, w, _, _ = precompute_lensing(phi_alm, nside, lmax,
                                         np.arange(hp.nside2npix(nside)))
        T_A = np.sum(w * T_map[nb], axis=0)
        sums["A"] += power(T_A, lh)

        # B: exact evaluation at the same deflected angles, nside = lmax.
        th, ph = deflected_angles(phi_alm, nside, lmax)
        sums["B"] += power(exact_eval(alm_T, lh, th, ph, nthreads), lh)

        # Bh / C: exact, on a fine grid (no aliasing), band-limited vs extended input.
        th_h, ph_h = deflected_angles(phi_alm, nside_hi, lmax)
        sums["Bh"] += power(exact_eval(alm_T, lh, th_h, ph_h, nthreads), lh)
        sums["C"] += power(exact_eval(alm_ext, lh_ext, th_h, ph_h, nthreads), lh)
        print(f"  lmax={lmax} realization {r + 1}/{n_real} done", flush=True)
    return {k: sums[k] / np.where(sums["unl"] > 0, sums["unl"], np.inf)
            for k in ("A", "B", "Bh", "C")}


def binned(ratio, lmax, weights):
    rows = []
    for lo, hi in BINS:
        hi = min(hi, lmax)
        if lo >= hi:
            rows.append(np.nan)
            continue
        ell = np.arange(lo, hi)
        wt = weights[ell]
        rows.append(float(np.sum(wt * (ratio[ell] - 1.0)) / np.sum(wt)) * 100.0)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lmax_list", type=int, nargs="+", default=[64, 128, 192])
    p.add_argument("--n_real", type=int, default=16)
    p.add_argument("--nside_hi_fac", type=int, default=4)
    p.add_argument("--nthreads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 8)))
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--out", default="results/analysis/lensing_power_transfer.npz")
    a = p.parse_args()

    lmax_top = 3 * max(a.lmax_list)
    cl_unl, cl_len, cl_pp = camb_spectra(lmax_top)
    ratio_D = np.where(cl_unl > 0, cl_len / np.where(cl_unl > 0, cl_unl, 1), 1.0)

    out = {"bins": np.array(BINS), "ratio_D": ratio_D, "cl_unl": cl_unl,
           "cl_len": cl_len, "cl_pp": cl_pp}
    print("\nBinned mean (ratio - 1) in %, weighted by (2l+1)"
          "   [A=diffcmb operator, B=exact same angles, Bh=exact no aliasing,"
          " C=exact unbandlimited input, D=CAMB theory]")
    for lmax in a.lmax_list:
        print(f"\nlmax = nside = {lmax}")
        res = run_lmax(lmax, a.n_real, cl_unl, cl_pp, a.nside_hi_fac, a.nthreads, a.seed)
        wts = 2.0 * np.arange(lmax) + 1.0
        tab = {k: binned(v, lmax, wts) for k, v in res.items()}
        tab["D"] = binned(ratio_D[:lmax], lmax, wts)
        print("  bin          " + "".join(f"{k:>9s}" for k in ("A", "B", "Bh", "C", "D")))
        for i, (lo, hi) in enumerate(BINS):
            if lo >= lmax:
                continue
            print(f"  [{lo:3d},{min(hi, lmax):3d})   "
                  + "".join(f"{tab[k][i]:9.3f}" for k in ("A", "B", "Bh", "C", "D")))
        for k, v in res.items():
            out[f"ratio_{k}_lmax{lmax}"] = v
        for k, v in tab.items():
            out[f"binned_{k}_lmax{lmax}"] = np.array(v)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, **out)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
