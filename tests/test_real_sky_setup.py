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
