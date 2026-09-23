from typing import List

import numpy as np

try:
    import camb
except Exception:  # keep module import-safe for environments without camb
    camb = None

try:
    import healpy as hp
except Exception:  # keep module import-safe for environments without healpy
    hp = None


def call_CAMB_map(_parameters: List[float], _lmax: int) -> np.ndarray:
    """Use CAMB to generate a power spectrum.

    If CAMB is not installed the function raises ImportError.
    """
    if camb is None:
        raise ImportError("camb is required for call_CAMB_map but is not installed")

    if _lmax <= 2551:
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=_parameters[0],
            ombh2=_parameters[1],
            omch2=_parameters[2],
            mnu=_parameters[3],
            omk=_parameters[4],
            tau=_parameters[5],
        )
        pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
        pars.set_for_lmax(_lmax, lens_potential_accuracy=0)

        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
        totCL = powers["total"]
        _DL = totCL[:, 0]

        _l = np.arange(len(_DL))
        _CL = []
        for i in range(_lmax):
            if i == 0:
                _CL.append(_DL[i])
            else:
                _CL.append(_DL[i] / (_l[i] * (_l[i] + 1)))

        return np.array(_CL)

    raise ValueError("lmax value is larger than the available data.")


def beam_pixwin_transfer(lmax: int, fwhm_arcmin: float, nside: int) -> np.ndarray:
    """Per-l amplitude transfer function B_l * pixwin_l for a Gaussian beam
    and the HEALPix pixel window, indexed l=0..lmax-1.

    Multiplies alm (not C_l) directly -- the diagonal harmonic-space forward
    operator ROADMAP.md Section 2 calls for as the beam/pixel-window
    pre-condition on real-data claims. `fwhm_arcmin=0.0` returns the pixel
    window alone (no beam smoothing).
    """
    if hp is None:
        raise ImportError("healpy is required for beam_pixwin_transfer")

    bl = hp.gauss_beam(fwhm=np.radians(fwhm_arcmin / 60.0), lmax=lmax - 1)
    pixwin = hp.pixwin(nside, lmax=lmax - 1)
    return bl * pixwin


def beam_pixwin_transfer_packed(lmax: int, fwhm_arcmin: float, nside: int) -> np.ndarray:
    """beam_pixwin_transfer broadcast onto the packed-alm index layout used
    throughout the sampler (samplers.py::_alm_index_lm: real parts L=2..lmax-1
    m=0..L, then imag parts L=2..lmax-1 m=2..L) -- ready to multiply directly
    against a packed alm vector as the diagonal beam/pixel-window operator.
    """
    from .alm_utils import packed_sizes
    from .samplers import _alm_index_lm

    n_real, n_imag = packed_sizes(lmax)
    L_arr, _m_arr = _alm_index_lm(lmax, n_real, n_imag)

    per_l = beam_pixwin_transfer(lmax, fwhm_arcmin, nside)
    return per_l[L_arr]


# ---------------------------------------------------------------------------
# Corrected fiducial (2026-09-23). Production scripts before this date used
# LCDM_PARAMS_LEGACY with call_CAMB_map, which has three defects, all
# self-consistent for sampler validation (the sampler learns whatever sky it is
# given) but wrong as a statement about LambdaCDM:
#   1. Omega_b = 0.0486 and Omega_c = 0.2589 were passed as ombh2/omch2
#      (physical densities), i.e. omega_b ~ 2.2x and omega_c ~ 2.2x too high;
#   2. call_CAMB_map converts D_l to C_l by 1/(l(l+1)) without the 2*pi,
#      so C_l^TT was 2*pi too small;
#   3. it returns CAMB's *lensed* total TT, which was then used as the
#      *unlensed* truth spectrum.
# Replays of old chains are unaffected: they read cl_true/cl_phiphi_true from
# the chain file, never recompute them.
# ---------------------------------------------------------------------------

LCDM_PARAMS_LEGACY = [67.74, 0.0486, 0.2589, 0.06, 0.0, 0.066]
_H = 0.6774
# [H0, ombh2, omch2, mnu, omk, tau]: the same Omega_b, Omega_c as the legacy
# list, now converted to the physical densities CAMB expects.
LCDM_PARAMS = [100.0 * _H, 0.0486 * _H ** 2, 0.2589 * _H ** 2, 0.06, 0.0, 0.066]


def fiducial_spectra(lmax: int, params=None):
    """(C_l^TT unlensed [muK^2], C_L^phiphi), indexed 0..lmax-1, zero below 2."""
    if camb is None:
        raise ImportError("camb is required for fiducial_spectra")
    p = LCDM_PARAMS if params is None else params
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=p[0], ombh2=p[1], omch2=p[2], mnu=p[3], omk=p[4], tau=p[5])
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(max(lmax + 100, 500), lens_potential_accuracy=1)
    res = camb.get_results(pars)
    tt = res.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)["unlensed_scalar"][:, 0]
    pp = res.get_lens_potential_cls(lmax=lmax - 1, raw_cl=True)[:, 0]
    cl_tt = np.zeros(lmax)
    cl_pp = np.zeros(lmax)
    cl_tt[2:] = tt[2:lmax]
    cl_pp[2:] = pp[2:lmax]
    return cl_tt, cl_pp
