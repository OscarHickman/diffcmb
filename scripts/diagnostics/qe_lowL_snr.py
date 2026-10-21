"""Low-L lensing information in temperature against the T multipole cap (ROADMAP T1.2b, T1.3:
"scope the scale first").

The TT quadratic estimator's Gaussian noise N_L (Okamoto & Hu; diffcmb.qe) for L <= LOUT only,
summing l1, l2 up to LMAX_T, so LMAX_T can reach the Planck range without qe.coupling_f_tt's
(lmax+1)^3 grid. Reports, per L-bin, N_L / C_L and the Fisher S/N on C_L^phiphi,
sqrt(sum_L f_sky (2L+1)/2 (C_L / (C_L + N_L))^2), for
  * this work's configuration: sigma = 30 muK per nside-64 pixel (1649 muK-arcmin), the
    5' beam and pixel window ignored (they only lower S/N), physical lensing (A_phi = 1) and
    the A_phi = 3000 amplification;
  * a Planck-like temperature map: 40 muK-arcmin white noise, 5' beam, l >= 100 as in the
    Planck 2018 lensing analysis (and l >= 2 for comparison).
The QE is the standard yardstick, not the joint posterior: at these S/N the joint posterior
cannot do much better (figure 3: about 0.6 of QE + prior at L ~ 15).

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/diagnostics/qe_lowL_snr.py [--lmax_t 64 128 300 1000 2000]
"""
import argparse

import numpy as np

from diffcmb import qe
from diffcmb.power import LCDM_PARAMS

LOUT = 63
BINS = ((2, 10), (10, 30), (30, 64))
ARCMIN = np.pi / 180 / 60


def lensed_spectra(lmax):
    """CAMB lensed C_l^TT [muK^2] and C_L^phiphi, l = 0..lmax, at the project's fiducial."""
    import camb

    p = LCDM_PARAMS
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=p[0], ombh2=p[1], omch2=p[2], mnu=p[3], omk=p[4], tau=p[5])
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(lmax + 500, lens_potential_accuracy=1)
    res = camb.get_results(pars)
    tt = res.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)["lensed_scalar"][: lmax + 1, 0]
    pp = res.get_lens_potential_cls(lmax=lmax, raw_cl=True)[:, 0]
    return tt, pp


def qe_noise_lowL(cl_lensed, cl_total, lmax_t, lout=LOUT):
    """N_L of the TT QE for L = 0..lout, with l1, l2 in [2, lmax_t] (inf where empty or L < 2).
    The same f^TT and weights as qe.qe_tt_noise_nl, restricted to the triads with L <= lout."""
    ell = np.arange(lmax_t + 1, dtype=np.float64)
    lap = ell * (ell + 1)
    inv = np.zeros(lmax_t + 1)
    good = cl_total[: lmax_t + 1] > 0
    inv[good] = 1 / cl_total[: lmax_t + 1][good]
    inv[:2] = 0
    L = np.arange(lout + 1, dtype=np.float64)
    fisher = np.zeros(lout + 1)
    for l1 in range(2, lmax_t + 1):
        for l2 in range(l1, min(l1 + lout, lmax_t) + 1):
            lmin, w = qe.wigner3j_000(l1, l2)
            n = min(lout + 1 - lmin, w.size)
            if n <= 0:
                continue
            Ls = np.arange(lmin, lmin + n)
            lapL = L[Ls] * (L[Ls] + 1)
            f = (np.sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * Ls + 1) / (16 * np.pi)) * w[:n]
                 * (cl_lensed[l1] * (lapL + lap[l1] - lap[l2]) + cl_lensed[l2] * (lapL + lap[l2] - lap[l1])))
            mult = 1.0 if l1 == l2 else 2.0  # the (l2, l1) term of the symmetric sum
            fisher[Ls] += mult * 0.5 * f ** 2 * inv[l1] * inv[l2]
    nl = np.full(lout + 1, np.inf)
    nz = fisher > 0
    nl[nz] = (2 * L[nz] + 1) / fisher[nz]
    nl[:2] = np.inf
    return nl


def bin_snr(cl_pp, nl, fsky):
    """Fisher S/N on C_L^phiphi per bin of BINS."""
    L = np.arange(nl.size)
    x = np.where(np.isfinite(nl), cl_pp[: nl.size] / (cl_pp[: nl.size] + nl), 0.0)
    return [np.sqrt(np.sum(fsky * (2 * L[lo:hi] + 1) / 2 * x[lo:hi] ** 2)) for lo, hi in BINS]


def white_noise(delta_t_arcmin, fwhm_arcmin, lmax):
    """Beam-deconvolved white noise C_l [muK^2]."""
    ell = np.arange(lmax + 1)
    sig = fwhm_arcmin * ARCMIN / np.sqrt(8 * np.log(2))
    return (delta_t_arcmin * ARCMIN) ** 2 * np.exp(ell * (ell + 1) * sig ** 2)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--lmax_t", type=int, nargs="+", default=[64, 128, 300, 1000, 2000])
    a = p.parse_args()
    top = max(a.lmax_t)
    tt, pp = lensed_spectra(top)
    pix_arcmin = np.sqrt(4 * np.pi / (12 * 64 ** 2)) / ARCMIN
    configs = [  # name, noise C_l, phi amplitude, f_sky, l_min of T
        ("this work, A_phi 1", white_noise(30 * pix_arcmin, 0, top), 1.0, 0.75, 2),
        ("this work, A_phi 3000", white_noise(30 * pix_arcmin, 0, top), 3000.0, 0.75, 2),
        ("Planck-like, l_T >= 2", white_noise(40, 5, top), 1.0, 0.67, 2),
        ("Planck-like, l_T >= 100", white_noise(40, 5, top), 1.0, 0.67, 100),
    ]
    print(f"this work's noise: 30 muK per {pix_arcmin:.1f}' pixel = {30 * pix_arcmin:.0f} muK-arcmin")
    hdr = "".join(f"  S/N {lo}-{hi - 1:<5}" for lo, hi in BINS)
    print(f"{'configuration':<26}{'lmax_T':>7}{hdr}  N/C at L=5, 20, 45")
    for name, nlt, amp, fsky, lmin in configs:
        for lmax_t in a.lmax_t:
            if name.startswith("this work") and lmax_t > 300:
                continue  # its noise swamps T above l ~ 100 anyway
            # the lensed T itself barely changes with A_phi at these l; the response scales with phi
            ctot = tt + nlt
            ctot[:lmin] = 0
            nl = qe_noise_lowL(tt, ctot, lmax_t)
            snr = bin_snr(amp * pp, nl, fsky)
            ratios = [nl[L] / (amp * pp[L]) for L in (5, 20, 45)]
            print(f"{name:<26}{lmax_t:>7}" + "".join(f"{s:>14.3g}" for s in snr)
                  + "   " + ", ".join(f"{r:.3g}" for r in ratios))


if __name__ == "__main__":
    main()
