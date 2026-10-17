"""scripts/real_sky_setup.py: mask loading and the ecliptic noise surrogate."""
import os
import sys

import numpy as np
import pytest

hp = pytest.importorskip("healpy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from real_sky_setup import ecliptic_noise_map, load_mask  # noqa: E402


def test_noise_is_lowest_at_the_ecliptic_poles_and_highest_on_the_plane():
    nside, sigma, amp = 16, 30.0, 0.5
    n = ecliptic_noise_map(nside, sigma, amp)
    rot = hp.Rotator(coord=["E", "G"])
    pole_pix = hp.ang2pix(nside, *rot(0.0, 0.0))
    plane_pix = hp.ang2pix(nside, *rot(np.pi / 2, 0.3))
    assert n[pole_pix] == pytest.approx(sigma, rel=0.02)
    assert n[plane_pix] == pytest.approx(sigma * (1 + amp), rel=0.02)
    assert n.min() >= sigma * 0.999 and n.max() <= sigma * (1 + amp) * 1.001


def test_zero_amplitude_is_uniform_and_negative_is_rejected():
    assert np.allclose(ecliptic_noise_map(8, 5.0, 0.0), 5.0)
    with pytest.raises(ValueError):
        ecliptic_noise_map(8, 5.0, -0.1)


def test_load_mask_downgrades_and_thresholds(tmp_path):
    hi = np.ones(hp.nside2npix(32))
    hi[hp.query_strip(32, 1.2, 1.9)] = 0.0  # equatorial band masked
    f = tmp_path / "m.fits"
    hp.write_map(str(f), hi, overwrite=True)
    keep = load_mask(str(f), 8)
    assert keep.dtype == bool and keep.size == hp.nside2npix(8)
    assert 0.3 < keep.mean() < 0.9
    th, _ = hp.pix2ang(8, np.where(~keep)[0])
    assert np.all(np.abs(th - np.pi / 2) < 0.6)


def test_load_mask_rejects_a_fully_masked_sky(tmp_path):
    f = tmp_path / "z.fits"
    hp.write_map(str(f), np.zeros(hp.nside2npix(8)), overwrite=True)
    with pytest.raises(ValueError):
        load_mask(str(f), 8)


def _bandlimited_sky(nside_in, lmax_in, fwhm_in_arcmin, seed=1):
    """A sky with power to lmax_in, observed through a Gaussian beam and the
    nside_in pixel window, and its unbeamed alm."""
    np.random.seed(seed)
    cl = 1.0 / (np.arange(lmax_in + 1) + 10.0) ** 2
    alm = hp.synalm(cl, lmax=lmax_in)
    t_in = hp.gauss_beam(np.radians(fwhm_in_arcmin / 60.0), lmax=lmax_in) * hp.pixwin(nside_in, lmax=lmax_in)
    return hp.alm2map(hp.almxfl(alm, t_in), nside_in, lmax=lmax_in), alm


def test_bandlimited_map_is_the_model_prediction_for_the_in_band_sky():
    from real_sky_setup import bandlimited_map
    nside_in, nside, lmax, fwhm_in = 128, 32, 24, 120.0
    sky, alm = _bandlimited_sky(nside_in, 3 * nside_in - 1, fwhm_in)
    transfer = hp.gauss_beam(np.radians(90.0 / 60.0), lmax=lmax - 1) * hp.pixwin(nside, lmax=lmax - 1)
    out = bandlimited_map(sky, nside, lmax, transfer, fwhm_in_arcmin=fwhm_in)
    a_ref = alm[hp.Alm.getidx(3 * nside_in - 1, *hp.Alm.getlm(lmax - 1))]
    expect = hp.alm2map(hp.almxfl(a_ref, np.r_[0.0, 0.0, transfer[2:]]), nside, lmax=lmax - 1)
    # No monopole or dipole (the sampler has none). The above-band power (l = 24..383, most of the variance) must not reach the output.
    assert np.std(out - expect) < 1e-3 * np.std(expect)


def test_bandlimited_map_rejects_a_transfer_of_the_wrong_length():
    from real_sky_setup import bandlimited_map
    sky = np.zeros(hp.nside2npix(16))
    with pytest.raises(ValueError):
        bandlimited_map(sky, 8, 12, np.ones(11))


def _write_data_map(path, nside=8, lmax=12, beam=5.0):
    m = np.arange(hp.nside2npix(nside), dtype=np.float64)
    hp.write_map(str(path), m, dtype=np.float64, overwrite=True,
                 extra_header=[("LMAX", lmax), ("BEAMFWHM", beam)])
    return m


def test_read_data_map_returns_the_map_when_the_header_matches(tmp_path):
    from real_sky_setup import read_data_map
    m = _write_data_map(tmp_path / "d.fits")
    assert np.array_equal(read_data_map(str(tmp_path / "d.fits"), 8, 12, 5.0), m)


@pytest.mark.parametrize("nside,lmax,beam", [(16, 12, 5.0), (8, 16, 5.0), (8, 12, 30.0), (8, 12, None)])
def test_read_data_map_refuses_a_map_prepared_for_another_model(tmp_path, nside, lmax, beam):
    from real_sky_setup import read_data_map
    _write_data_map(tmp_path / "d.fits")
    with pytest.raises(ValueError):
        read_data_map(str(tmp_path / "d.fits"), nside, lmax, beam)
