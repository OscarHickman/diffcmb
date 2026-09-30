"""--cl_init data helper (ROADMAP T0.1 pilot): S_l/(2l-3) from packed alm."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from coverage_ensemble_chain import data_driven_ln_cl  # noqa: E402


def _pack(alm_by_l, lmax):
    re, im = [], []
    for L in range(2, lmax):
        re.extend(alm_by_l[L].real[: L + 1])
    for L in range(2, lmax):
        im.extend(alm_by_l[L].imag[1: L + 1])
    return np.array(re + im)


def test_data_driven_ln_cl_matches_direct_power():
    lmax = 6
    rng = np.random.default_rng(0)
    alm = {L: rng.normal(size=L + 1) + 1j * rng.normal(size=L + 1) for L in range(2, lmax)}
    for L in alm:
        alm[L][0] = alm[L][0].real
    got = np.exp(data_driven_ln_cl(_pack(alm, lmax), lmax))
    for L in range(2, lmax):
        s = abs(alm[L][0]) ** 2 + 2 * np.sum(np.abs(alm[L][1:]) ** 2)
        assert np.isclose(got[L - 2], s / (2 * L + 1 - 4))


def test_data_driven_ln_cl_finite_for_zero_alm():
    lmax = 5
    n = sum(L + 1 for L in range(2, lmax)) + sum(L for L in range(2, lmax))
    assert np.all(np.isfinite(data_driven_ln_cl(np.zeros(n), lmax)))
