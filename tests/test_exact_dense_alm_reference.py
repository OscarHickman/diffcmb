"""scripts/exact_dense_alm_reference.py (ROADMAP T0.1, 2026-09-27).

The dense reference is only a reference if its draws are exact: the a | C
draw must have mean P^-1 A^T d / sigma^2 and covariance P^-1, and the C_l
draw must be Block 1's flat-prior conditional.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import exact_dense_alm_reference as edr  # noqa: E402
from null_alm_power_rank_flat_prior import packed_layout  # noqa: E402

LMAX = 6


def _problem(seed=0):
    rng = np.random.default_rng(seed)
    L_arr, v = packed_layout(LMAX)
    A = rng.normal(size=(60, L_arr.size))
    C = np.zeros(LMAX)
    C[2:] = 1.0 / np.arange(2, LMAX)
    d = rng.normal(size=60)
    return A, C, d, L_arr, v, rng


def test_alm_draw_has_the_exact_gaussian_mean_and_covariance():
    A, C, d, L_arr, v, rng = _problem()
    sigma2 = 0.7
    P = A.T @ A / sigma2 + np.diag(1.0 / (C[L_arr] * v))
    cov = np.linalg.inv(P)
    mean = cov @ (A.T @ d / sigma2)
    AtA, Atd = A.T @ A, A.T @ d
    draws = np.array([edr.draw_alm_given_cl(AtA.copy(), Atd, sigma2, C, L_arr, v, rng)
                      for _ in range(20000)])
    assert np.allclose(draws.mean(0), mean, atol=5 * np.sqrt(np.diag(cov).max() / 20000))
    emp = np.cov(draws.T)
    assert np.allclose(emp, cov, atol=0.05 * np.abs(cov).max())


def test_alm_draw_does_not_modify_the_cached_normal_matrix():
    A, C, d, L_arr, v, rng = _problem(1)
    AtA = A.T @ A
    before = AtA.copy()
    edr.draw_alm_given_cl(AtA, A.T @ d, 1.0, C, L_arr, v, rng)
    assert np.array_equal(AtA, before)


def test_cl_draws_follow_the_flat_prior_conditional_when_data_pin_the_alm():
    # sigma -> 0: a is pinned at the data, so C_l | a is the Block 1 InvGamma
    A = np.eye(packed_layout(LMAX)[0].size)
    L_arr, v = packed_layout(LMAX)
    rng = np.random.default_rng(3)
    a_true = rng.normal(size=L_arr.size) * np.sqrt(v)
    _, c_s = edr.exact_gibbs(A, A.T @ a_true, 1e-10, LMAX, 4000, 10, 1, rng,
                             np.ones(LMAX))
    S = np.bincount(L_arr, weights=a_true ** 2 / v, minlength=LMAX)
    shape = edr.invgamma_shape_for_spectrum(LMAX, a0=-1.0)
    from scipy.stats import invgamma

    for ell in range(2, LMAX):
        # the median: at l = 2 the shape is 1.5 and the variance is infinite
        expect = invgamma.median(shape[ell], scale=S[ell] / 2.0)
        assert np.median(c_s[:, ell - 2]) == pytest.approx(expect, rel=0.06)
