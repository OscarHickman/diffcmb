"""scripts/null_alm_power_rank_flat_prior.py (ROADMAP T0.1c).

The null must be exact, since it decides whether the a_lm `[30,60)` offset is
the statistic or the sampler. Tests: the flat-prior C_l marginal against grid
quadrature; the high-S/N limit (data pin a_lm, so ranks are uniform); the
noise-dominated limit (the flat prior inflates posterior power, truth ranks
low); and recovery of a known posterior-variance fraction from chains.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("scripts", os.path.join("scripts", "paper")):
    path = os.path.join(REPO, sub)
    if path not in sys.path:
        sys.path.insert(0, path)

import null_alm_power_rank_flat_prior as nul  # noqa: E402

from diffcmb.alm_utils import packed_dof_per_multipole  # noqa: E402

LMAX = 12


def _cl():
    ell = np.arange(LMAX, dtype=np.float64)
    cl = np.zeros(LMAX)
    cl[2:] = 1.0 / (ell[2:] * (ell[2:] + 1.0))
    return cl


def test_packed_layout_weights_match_dof():
    L_arr, v = nul.packed_layout(LMAX)
    dof = np.bincount(L_arr, minlength=LMAX)
    assert np.array_equal(dof, packed_dof_per_multipole(LMAX))
    # m=0 carries weight 1, each of the 2L real/imag parts at m>0 weight 1/2
    for L in range(2, LMAX):
        assert v[L_arr == L].sum() == pytest.approx(1.0 + 0.5 * 2 * L)


@pytest.mark.parametrize("noise_ratio", [0.5, 5.0])
def test_C_marginal_matches_grid_quadrature(noise_ratio):
    # noise_ratio 5 is where the X > N_l truncation carries real mass
    rng = np.random.default_rng(0)
    cl, L = _cl(), 5
    noise = np.full(LMAX, noise_ratio * cl[L])
    L_arr, v = nul.packed_layout(LMAX)
    d = np.sqrt((cl[L_arr] + noise[L_arr]) * v) * rng.standard_normal(L_arr.size)
    n_draws = 40000
    post = nul.exact_posterior_draws(d, noise, LMAX, n_draws, rng)
    # E[a_j^2 | d] summed over l=L, against quadrature over the C marginal
    k = int(packed_dof_per_multipole(LMAX)[L])
    D = np.sum(d[L_arr == L] ** 2 / v[L_arr == L])
    C = np.linspace(1e-6, 300 * cl[L], 1000000)
    X = C + noise[L]
    logp = -0.5 * k * np.log(X) - D / (2 * X)
    p = np.exp(logp - logp.max())
    W = C / X
    sel = L_arr == L
    e_a2 = np.sum((W[:, None] ** 2 * d[sel] ** 2
                   + W[:, None] * noise[L] * v[sel]) * p[:, None], axis=0) / p.sum()
    got = np.mean(post[:, sel] ** 2, axis=0)
    assert np.allclose(got, e_a2, rtol=0.05)


def test_high_snr_null_is_uniform():
    cl = _cl()
    noise = np.zeros(LMAX)
    noise[2:] = 1e-6 * cl[2:]
    ranks = nul.null_ranks(cl, noise, LMAX, n_null=400, n_draws=60, seed=1)
    for r in ranks.values():
        u = (r + 0.5) / 61.0
        assert abs(u.mean() - 0.5) < 4 * 0.289 / np.sqrt(len(u))


def test_noise_dominated_flat_prior_makes_truth_rank_low():
    cl = _cl()
    noise = np.zeros(LMAX)
    noise[2:] = 3.0 * cl[2:]
    ranks = nul.null_ranks(cl, noise, LMAX, n_null=400, n_draws=60, seed=2)
    u = (ranks[(2, 10)] + 0.5) / 61.0
    assert u.mean() < 0.5 - 4 * 0.289 / np.sqrt(len(u))


def test_effective_noise_recovers_known_variance_fraction(tmp_path):
    rng = np.random.default_rng(3)
    cl = _cl()
    L_arr, v = nul.packed_layout(LMAX)
    w_true = np.linspace(0.2, 0.9, LMAX)
    for c in range(6):
        samp = np.sqrt(w_true[L_arr] * cl[L_arr] * v) * rng.standard_normal(
            (4000, L_arr.size))
        np.savez(tmp_path / f"chain_r{c:03d}.npz", lmax=LMAX, cl_true=cl,
                 alm_samples=np.hstack([np.zeros((4000, LMAX - 2)), samp]))
    files = sorted(str(p) for p in tmp_path.glob("chain_r*.npz"))
    n_eff, w = nul.effective_noise(files, LMAX)
    assert np.allclose(w[2:], w_true[2:], rtol=0.03)
    assert np.allclose(n_eff[2:], cl[2:] * (1 - w_true[2:]) / w_true[2:], rtol=0.1)
