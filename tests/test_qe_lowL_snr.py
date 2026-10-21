"""scripts/diagnostics/qe_lowL_snr.py: the L <= LOUT quadratic-estimator noise matches diffcmb.qe."""
import numpy as np
import qe_lowL_snr as q

from diffcmb import qe


def _spectra(lmax):
    ell = np.arange(lmax + 1, dtype=float)
    tt = np.r_[0, 0, 2 * np.pi * 1000.0 / (ell[2:] * (ell[2:] + 1))]
    return tt, tt + 50.0 * np.exp(ell / 40.0)


def test_low_L_noise_matches_the_full_grid_estimator():
    lmax = 40
    tt, tot = _spectra(lmax)
    full = qe.qe_tt_noise_nl(tt, tot, lmax)
    low = q.qe_noise_lowL(tt, tot, lmax, lout=25)
    np.testing.assert_allclose(low[2:26], full[2:26], rtol=1e-10)
    assert np.all(np.isinf(low[:2]))


def test_snr_grows_with_the_T_cap_and_vanishes_for_huge_noise():
    tt, tot = _spectra(120)
    pp = np.r_[0, 0, 1e-7 / np.arange(2, 121) ** 4]
    a = q.bin_snr(pp, q.qe_noise_lowL(tt, tot, 60, lout=40), 1.0)
    b = q.bin_snr(pp, q.qe_noise_lowL(tt, tot, 120, lout=40), 1.0)
    assert all(y > x for x, y in zip(a, b) if x > 0)
    tiny = q.bin_snr(pp, q.qe_noise_lowL(tt, tot * 1e6, 60, lout=40), 1.0)
    assert max(tiny) < 1e-6
