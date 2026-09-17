"""Tests for the curved-sky TT quadratic-estimator lensing noise curve.

These are the fast, deterministic checks (project convention: `tests/` is
small-lmax pytest, `scripts/` carries the production validation). The
*physics* validation of N_L -- agreement with the autodiff/FD Fisher
curvature of the real forward model -- lives in
`scripts/validate_qe_noise.py`, because it needs a built model and is far
too slow for the unit suite.
"""

import numpy as np
import pytest

from diffcmb import qe


def test_wigner3j_matches_sympy_exactly():
    """ducc0's Schulten-Gordon recursion vs sympy's exact rationals.

    Everything in this module rests on the 3j symbols, so they are checked
    against an independent exact implementation rather than assumed.
    """
    sympy_wigner = pytest.importorskip("sympy.physics.wigner")
    rng = np.random.default_rng(0)
    for _ in range(6):
        l2 = int(rng.integers(0, 24))
        l3 = int(rng.integers(0, 24))
        lmin, arr = qe.wigner3j_000(l2, l3)
        for i, got in enumerate(arr):
            ref = float(sympy_wigner.wigner_3j(lmin + i, l2, l3, 0, 0, 0))
            assert got == pytest.approx(ref, abs=1e-12)


def test_coupling_f_is_symmetric_under_l1_l2_swap():
    """f^TT_{l1 l2 L} must be symmetric: it multiplies T_{l1m1}T_{l2m2}, which is."""
    lmax = 24
    cl = _toy_cl(lmax)
    f = qe.coupling_f_tt(cl, lmax)
    assert np.allclose(f, np.swapaxes(f, 0, 1), rtol=0, atol=1e-13)


def test_coupling_f_vanishes_on_parity_forbidden_triads():
    """The (l1 l2 L; 0 0 0) symbol is zero unless l1+l2+L is even."""
    lmax = 16
    f = qe.coupling_f_tt(_toy_cl(lmax), lmax)
    l1, l2, L = np.meshgrid(np.arange(lmax + 1), np.arange(lmax + 1),
                            np.arange(lmax + 1), indexing="ij")
    odd = ((l1 + l2 + L) % 2) == 1
    assert np.all(f[odd] == 0.0)


def test_coupling_f_vanishes_outside_triangle():
    lmax = 16
    f = qe.coupling_f_tt(_toy_cl(lmax), lmax)
    l1, l2, L = np.meshgrid(np.arange(lmax + 1), np.arange(lmax + 1),
                            np.arange(lmax + 1), indexing="ij")
    outside = (L < np.abs(l1 - l2)) | (L > l1 + l2)
    assert np.all(f[outside] == 0.0)


def test_nl_is_positive_and_finite_where_defined():
    lmax = 32
    cl = _toy_cl(lmax)
    nl = qe.qe_tt_noise_nl(cl, cl + _toy_noise(lmax), lmax)
    assert np.all(np.isfinite(nl[2:]))
    assert np.all(nl[2:] > 0.0)


def test_nl_decreases_when_instrument_noise_decreases():
    """Less instrumental noise must never make the reconstruction worse."""
    lmax = 32
    cl = _toy_cl(lmax)
    noisy = qe.qe_tt_noise_nl(cl, cl + _toy_noise(lmax, amp=4.0), lmax)
    clean = qe.qe_tt_noise_nl(cl, cl + _toy_noise(lmax, amp=1.0), lmax)
    assert np.all(clean[2:] < noisy[2:])


def test_nl_scales_as_inverse_square_of_cmb_gradient_power():
    """N_L ~ 1/f^2 and f ~ C_l, so scaling the *signal* that sources the
    coupling (holding the total filter fixed) must scale N_L as 1/a^2."""
    lmax = 24
    cl = _toy_cl(lmax)
    tot = cl + _toy_noise(lmax)
    base = qe.qe_tt_noise_nl(cl, tot, lmax)
    scaled = qe.qe_tt_noise_nl(3.0 * cl, tot, lmax)
    assert np.allclose(scaled[2:], base[2:] / 9.0, rtol=1e-10)


def test_noise_cl_from_pixel_sigma_matches_white_normalisation():
    """N_l = sigma_pix^2 * Omega_pix, flat in l."""
    nside = 16
    npix = 12 * nside ** 2
    nl = qe.white_noise_cl(sigma_pixel=2.0, npix=npix, lmax=8)
    assert nl.shape == (9,)
    assert np.allclose(nl, 4.0 * 4.0 * np.pi / npix)


def test_rejects_short_spectrum():
    with pytest.raises(ValueError, match="lmax"):
        qe.qe_tt_noise_nl(np.ones(5), np.ones(5), lmax=32)


def _toy_cl(lmax):
    ell = np.arange(lmax + 1, dtype=np.float64)
    cl = np.zeros(lmax + 1)
    cl[2:] = 1.0 / (ell[2:] * (ell[2:] + 1.0))
    return cl


def _toy_noise(lmax, amp=1.0):
    return np.full(lmax + 1, amp * 1e-3)
