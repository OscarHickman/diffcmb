"""scripts/compare_band_edge_paired.py (ROADMAP T0.1 a, 2026-10-07).

The T0.1 a harvest compares posterior C_l / (S_true/(k-4)) at the band edge
between ensembles run on the SAME skies (the phi-fixed run, the production
control, the exact dense reference), paired by sky. The comparison is only
valid if the skies really match, so it must refuse unmatched truths, pair on
the realizations every directory has, and read a shifted sampler as shifted
and an exact one as consistent.
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

import compare_band_edge_paired as cmp  # noqa: E402
import diagnose_alm_band_edge as dbe  # noqa: E402
import null_alm_power_rank_flat_prior as nul  # noqa: E402

from diffcmb.alm_utils import packed_dof_per_multipole  # noqa: E402

LMAX = 12
N_SKIES = 12
N_DRAWS = 200


def _skies(seed=0):
    ell = np.arange(LMAX, dtype=np.float64)
    cl = np.zeros(LMAX)
    cl[2:] = 1.0 / (ell[2:] * (ell[2:] + 1.0))
    return dbe.null_pairs(cl, 0.5 * cl, LMAX, N_SKIES, seed)


def _write(d, skies, seed, top_shift=0.0, realizations=None):
    """Exact flat-prior (C, a) draws per sky; top_shift moves ln C at l = lmax-1."""
    os.makedirs(d, exist_ok=True)
    L_arr, v = nul.packed_layout(LMAX)
    k = packed_dof_per_multipole(LMAX)
    for i, (draws, truth) in enumerate(skies):
        r = i if realizations is None else realizations[i]
        rng = np.random.default_rng(seed * 1000 + i)
        a = draws[rng.permutation(draws.shape[0])][:N_DRAWS]
        S = np.stack([np.bincount(L_arr, weights=x ** 2 / v, minlength=LMAX) for x in a])
        C = S[:, 2:] / 2.0 / rng.gamma(k[2:] / 2.0 - 1.0, size=S[:, 2:].shape)
        lnc = np.log(C)
        lnc[:, -1] += top_shift
        np.savez(os.path.join(d, f"chain_r{r:03d}.npz"),
                 alm_samples=np.concatenate([lnc, a], axis=1), alm_true_packed=truth,
                 lmax=LMAX, realization=r)


def test_exact_runs_agree_and_a_shifted_top_multipole_is_flagged(tmp_path):
    skies = _skies()
    ref, good, low = (str(tmp_path / n) for n in ("ref", "good", "low"))
    _write(ref, skies, seed=1)
    _write(good, skies, seed=2)
    # the same draws as `good`, with C at l = lmax-1 scaled by 0.9: the only
    # difference between the two runs is the shift, so the test is exact
    _write(low, skies, seed=2, top_shift=np.log(0.9))
    res = cmp.compare(ref, [good, low], exclude=())
    top = LMAX - 1
    g, lo = res["paired"][good], res["paired"][low]
    assert abs(g["diff"][top] / g["se"][top]) < 3.0
    assert lo["diff"][top] / lo["se"][top] < -3.0
    np.testing.assert_allclose(lo["diff"][top] - g["diff"][top],
                               -0.1 * res["ratio"][good][:, -1, top].mean(), rtol=1e-10)
    # the shift lands on l = lmax-1 and nowhere else
    np.testing.assert_array_equal(lo["diff"][:top], g["diff"][:top])
    assert res["skies"] == list(range(N_SKIES))


def test_pairs_on_common_skies_and_honours_exclude(tmp_path):
    skies = _skies()
    ref, run = str(tmp_path / "ref"), str(tmp_path / "run")
    _write(ref, skies, seed=1)
    _write(run, skies[:8], seed=2)
    res = cmp.compare(ref, [run], exclude=(3,))
    assert res["skies"] == [0, 1, 2, 4, 5, 6, 7]


def test_refuses_skies_whose_truths_differ(tmp_path):
    ref, run = str(tmp_path / "ref"), str(tmp_path / "run")
    _write(ref, _skies(seed=0), seed=1)
    _write(run, _skies(seed=9), seed=2)
    with pytest.raises(SystemExit, match="not the same sky"):
        cmp.compare(ref, [run], exclude=())


def test_per_chain_ratio_averages_to_the_ensemble_ratio(tmp_path):
    skies = _skies()
    d = str(tmp_path / "ens")
    _write(d, skies, seed=4)
    files = dbe.chain_files(d)
    per = np.mean([dbe.cl_ratio_per_chain(f, LMAX) for f in files], axis=0)
    np.testing.assert_allclose(per, dbe.cl_ratio_by_quarter(files, LMAX), rtol=1e-12)
