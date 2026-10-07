"""strict_sbc_ranks (ROADMAP T1.1 / T0.1 c): strict SBC of the spectra in the
strict exact ensemble, plus the C_l collapse table.

Synthetic chain files exercise the statistics (calibrated passes, a shifted
posterior fails, collapse is flagged, non-strict chains are refused); one real
toy-size strict chain checks the script reads production files.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
pytest.importorskip("healpy")

from diffcmb.alm_utils import packed_length  # noqa: E402

LMAX = 8
NL = LMAX - 2
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fake_chain(path, r, rng, n=400, shift=0.0, collapse_ell=None, strict=True):
    """Posterior = prior case: truth and draws are independent N(0, 0.3) in ln C
    around ln 1, so strict ranks are exactly uniform unless `shift` moves the draws."""
    ln_tt_true, ln_pp_true = rng.normal(0, 0.3, NL), rng.normal(0, 0.3, NL)
    ln_tt = rng.normal(0, 0.3, (n, NL)) + shift
    ln_pp = rng.normal(0, 0.3, (n, NL)) + shift
    alm_true = rng.normal(size=packed_length(LMAX))
    if collapse_ell is not None:
        ln_tt[: n // 2, collapse_ell - 2] = np.log(1e-6)
    cl_true = np.zeros(LMAX)
    cl_true[2:] = np.exp(ln_tt_true)
    clpp_true = np.zeros(LMAX)
    clpp_true[2:] = np.exp(ln_pp_true)
    np.savez(path, realization=r, lmax=LMAX,
             alm_samples=np.concatenate([ln_tt, rng.normal(size=(n, packed_length(LMAX)))], 1),
             cl_phiphi_samples=ln_pp, cl_true=cl_true, cl_phiphi_true=clpp_true,
             alm_true_packed=alm_true, cl_prior_nu=30.0 if strict else np.nan,
             cl_phiphi_prior_nu=30.0)


def _ensemble(tmp_path, n_chains, seed, **kw):
    rng = np.random.default_rng(seed)
    for r in range(n_chains):
        _fake_chain(tmp_path / f"chain_r{r:03d}.npz", r, rng, **kw)
    return sorted(tmp_path.glob("chain_r*.npz"))


def test_calibrated_ensemble_passes(tmp_path):
    import strict_sbc_ranks as ssr

    res = ssr.strict_ranks(_ensemble(tmp_path, 200, 0), thin=10, burn_frac=0.0)
    assert res["n_draws"] == 40
    for block in ("C_l^TT", "C_L^phiphi"):
        assert res["ranks"][block].shape == (200, 1 + 1)  # bin [2,8) plus the l = lmax-1 row
        assert np.all(res["p"][block] > 0.001), res["p"][block]


def test_shifted_posterior_fails(tmp_path):
    import strict_sbc_ranks as ssr

    res = ssr.strict_ranks(_ensemble(tmp_path, 200, 1, shift=0.3), thin=10, burn_frac=0.0)
    assert np.all(res["p"]["C_l^TT"] < 0.001) and np.all(res["p"]["C_L^phiphi"] < 0.001)


def test_thinning_and_burn_in_set_the_draw_count(tmp_path):
    import strict_sbc_ranks as ssr

    res = ssr.strict_ranks(_ensemble(tmp_path, 3, 2), thin=7, burn_frac=0.5)
    assert res["n_draws"] == len(range(200, 400, 7))


def test_refuses_a_non_strict_chain(tmp_path):
    import strict_sbc_ranks as ssr

    files = _ensemble(tmp_path, 3, 3)
    _fake_chain(tmp_path / "chain_r099.npz", 99, np.random.default_rng(9), strict=False)
    with pytest.raises(SystemExit, match="cl_prior_nu"):
        ssr.strict_ranks(files + [tmp_path / "chain_r099.npz"], thin=10, burn_frac=0.0)


def test_collapse_is_flagged_by_quarter(tmp_path):
    import strict_sbc_ranks as ssr

    rng = np.random.default_rng(4)
    _fake_chain(tmp_path / "chain_r000.npz", 0, rng)
    _fake_chain(tmp_path / "chain_r001.npz", 1, rng, collapse_ell=6)
    rows = ssr.collapse_table(sorted(tmp_path.glob("chain_r*.npz")))
    assert [(r["realization"], r["ell"]) for r in rows] == [(1, 6)]
    assert rows[0]["quarters_collapsed"] == [1, 2]


def test_cli_runs_on_a_real_strict_chain(tmp_path):
    from test_scripts_pipeline import _run_chain

    for r in (1, 2):
        _run_chain(tmp_path, "--cl_prior_nu", "30", "--cl_phiphi_prior_nu", "30", realization=r)
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "diffcmb"))
    res = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "strict_sbc_ranks.py"),
                          "--indir", str(tmp_path), "--thin", "1", "--out", str(tmp_path / "s.npz")],
                         env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    s = np.load(tmp_path / "s.npz")
    assert s["ranks_C_l^TT"].shape == (2, 2) and int(s["n_draws"]) == 6
    assert "collapse" in res.stdout
