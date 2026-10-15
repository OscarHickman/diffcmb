"""npe_compress (ROADMAP T1.1, width-gap route b): the learned map compressor.

A network from the data a_lm (plus the hand summaries x) to a point estimate of
theta, whose output joins x as the flow's conditioning. Its map branch is
quadratic (convolution over the (l, m) grid, then a square), so it can learn
power and, unlike a power spectrum, couplings between neighbouring multipoles,
which is where lensing puts its information. Toy problems pin both abilities.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("tensorflow")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from diffcmb.alm_utils import packed_length, packed_sizes  # noqa: E402
from diffcmb.samplers import _alm_index_lm  # noqa: E402

LMAX = 10
NPK = packed_length(LMAX)
KW = {"epochs": 25, "batch_size": 128, "learning_rate": 2e-3, "seed": 0, "verbose": False,
      "conv_filters": 8}


def _lm():
    return _alm_index_lm(LMAX, *packed_sizes(LMAX))


def _power_problem(n, seed):
    """theta = ln A; the a_lm at l >= 6 have variance A, the rest unit; x is noise."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-1.0, 1.0, size=(n, 1))
    L, _ = _lm()
    alm = rng.normal(size=(n, NPK))
    alm[:, L >= 6] *= np.exp(0.5 * theta)
    return alm, rng.normal(size=(n, 3)), theta


def _coupling_problem(n, seed):
    """theta = rho; a_{l+1,m} = rho a_{l,m} + sqrt(1 - rho^2) e for even l, so every
    multipole keeps unit variance: power carries no information, the coupling does."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-0.8, 0.8, size=(n, 1))
    L, m = _lm()
    n_real = packed_sizes(LMAX)[0]
    part = (np.arange(NPK) >= n_real).astype(int)
    alm = rng.normal(size=(n, NPK))
    pos = {(int(a), int(b), int(c)): i for i, (a, b, c) in enumerate(zip(L, m, part))}
    for (ell, mm, pp), i in pos.items():
        j = pos.get((ell + 1, mm, pp))
        if ell % 2 == 0 and j is not None:
            alm[:, j] = theta[:, 0] * alm[:, i] + np.sqrt(1 - theta[:, 0] ** 2) * alm[:, j]
    return alm, rng.normal(size=(n, 3)), theta


def _corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def test_map_branch_learns_power():
    import npe_compress as nc

    comp = nc.MapCompressor.fit(*_power_problem(6000, 0), lmax=LMAX, **KW)
    alm, x, theta = _power_problem(1000, 1)
    assert _corr(comp.predict(alm, x), theta) > 0.9


def test_map_branch_learns_a_coupling_that_power_cannot_see():
    import npe_compress as nc

    alm, x, theta = _coupling_problem(1000, 3)
    L, _ = _lm()
    power = np.stack([np.mean(alm[:, L == ell] ** 2, axis=1) for ell in range(2, LMAX)], 1)
    assert max(abs(_corr(power[:, k], theta)) for k in range(power.shape[1])) < 0.2
    comp = nc.MapCompressor.fit(*_coupling_problem(8000, 2), lmax=LMAX, **KW)
    assert _corr(comp.predict(alm, x), theta) > 0.8


def test_save_load_round_trip(tmp_path):
    import npe_compress as nc

    alm, x, theta = _power_problem(600, 4)
    comp = nc.MapCompressor.fit(alm, x, theta, lmax=LMAX, **{**KW, "epochs": 2})
    comp.save(tmp_path / "c")
    again = nc.MapCompressor.load(tmp_path / "c")
    np.testing.assert_allclose(again.predict(alm[:50], x[:50]), comp.predict(alm[:50], x[:50]),
                               rtol=1e-5, atol=1e-6)


def test_conditioning_appends_the_compressor_output_or_passes_x_through(tmp_path):
    import npe_compress as nc

    alm, x, theta = _power_problem(400, 5)
    comp = nc.MapCompressor.fit(alm, x, theta, lmax=LMAX, **{**KW, "epochs": 1})
    comp.save(tmp_path / "c")
    for name, cfg in (("with", {"compressor": str(tmp_path / "c")}), ("without", {})):
        (tmp_path / name).mkdir()
        (tmp_path / name / "sim_config.json").write_text(json.dumps(cfg))
    out = nc.load_conditioning(tmp_path / "with")(x[:5], alm[:5])
    np.testing.assert_allclose(out, np.concatenate([x[:5], comp.predict(alm[:5], x[:5])], 1),
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(nc.load_conditioning(tmp_path / "without")(x[:5], None), x[:5])
    with pytest.raises(ValueError, match="data a_lm"):
        nc.load_conditioning(tmp_path / "with")(x[:5], None)
