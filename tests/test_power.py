import numpy as np
import pytest

from diffcmb import power
from diffcmb.alm_utils import packed_sizes

try:
    import healpy as hp
except Exception:
    hp = None


def test_call_camb_map_requires_camb():
    # If camb isn't installed the function should raise ImportError
    try:
        camb_installed = True
    except Exception:
        camb_installed = False

    if not camb_installed:
        with pytest.raises(ImportError):
            power.call_CAMB_map([67.0, 0.022, 0.12, 0.06, 0.0, 0.06], 10)
    else:
        # Basic smoke test when camb is available: returns array of length lmax
        out = power.call_CAMB_map([67.0, 0.022, 0.12, 0.06, 0.0, 0.06], 10)
        assert len(out) == 10


def test_beam_pixwin_transfer_requires_healpy():
    if hp is not None:
        pytest.skip("healpy is installed; ImportError path not exercised here")
    with pytest.raises(ImportError):
        power.beam_pixwin_transfer(lmax=10, fwhm_arcmin=5.0, nside=4)


@pytest.mark.skipif(hp is None, reason="healpy not installed")
def test_beam_pixwin_transfer_matches_healpy_directly():
    lmax, fwhm_arcmin, nside = 20, 5.0, 8
    out = power.beam_pixwin_transfer(lmax=lmax, fwhm_arcmin=fwhm_arcmin, nside=nside)

    expected_bl = hp.gauss_beam(fwhm=np.radians(fwhm_arcmin / 60.0), lmax=lmax - 1)
    expected_pixwin = hp.pixwin(nside, lmax=lmax - 1)
    np.testing.assert_allclose(out, expected_bl * expected_pixwin)


@pytest.mark.skipif(hp is None, reason="healpy not installed")
def test_beam_pixwin_transfer_length_matches_lmax():
    lmax = 30
    out = power.beam_pixwin_transfer(lmax=lmax, fwhm_arcmin=10.0, nside=16)
    assert len(out) == lmax


@pytest.mark.skipif(hp is None, reason="healpy not installed")
def test_beam_pixwin_transfer_decreases_with_ell():
    # A Gaussian beam + pixel window is a low-pass filter: monotonically
    # non-increasing in l is the defining physical property being encoded.
    lmax = 50
    out = power.beam_pixwin_transfer(lmax=lmax, fwhm_arcmin=10.0, nside=32)
    assert np.all(np.diff(out) <= 1e-12)


@pytest.mark.skipif(hp is None, reason="healpy not installed")
def test_beam_pixwin_transfer_zero_fwhm_is_pixwin_only():
    lmax, nside = 20, 8
    out = power.beam_pixwin_transfer(lmax=lmax, fwhm_arcmin=0.0, nside=nside)
    expected_pixwin = hp.pixwin(nside, lmax=lmax - 1)
    np.testing.assert_allclose(out, expected_pixwin)


@pytest.mark.skipif(hp is None, reason="healpy not installed")
def test_beam_pixwin_transfer_packed_length_matches_alm_layout():
    lmax = 20
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    out = power.beam_pixwin_transfer_packed(lmax=lmax, fwhm_arcmin=5.0, nside=8)
    assert len(out) == n_real + n_imag


@pytest.mark.skipif(hp is None, reason="healpy not installed")
def test_beam_pixwin_transfer_packed_matches_per_l_broadcast():
    # Every packed-alm entry for a given l must carry that l's per-l transfer
    # value -- i.e. the packed array is exactly per_l[L_arr], not some other
    # ordering or scaling.
    from diffcmb.samplers import _alm_index_lm

    lmax = 20
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    L_arr, _m_arr = _alm_index_lm(lmax, n_real, n_imag)

    per_l = power.beam_pixwin_transfer(lmax=lmax, fwhm_arcmin=5.0, nside=8)
    packed = power.beam_pixwin_transfer_packed(lmax=lmax, fwhm_arcmin=5.0, nside=8)

    np.testing.assert_allclose(packed, per_l[L_arr])
