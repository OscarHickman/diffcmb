"""scripts/diagnose_alm_band_edge.py (ROADMAP T0.1, 2026-09-27).

The diagnostic decides whether the top a_lm bin's power-rank offset is a
sampler defect, so it must read an exact sampler as calibrated and flag a
sampler whose posterior a_lm power is shrunk, with the sign seen at the band
edge (truth ranking high, slope below W, sd(z) above 1).
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

import diagnose_alm_band_edge as dbe  # noqa: E402
import null_alm_power_rank_flat_prior as nul  # noqa: E402

LMAX = 12
N_CHAINS = 40
N_DRAWS = 60


def _cl():
    ell = np.arange(LMAX, dtype=np.float64)
    cl = np.zeros(LMAX)
    cl[2:] = 1.0 / (ell[2:] * (ell[2:] + 1.0))
    return cl


def _ensemble(shrink=1.0, seed=0):
    """(chains, noise): exact flat-prior posterior draws, optionally shrunk."""
    cl = _cl()
    noise = 0.5 * cl
    pairs = dbe.null_pairs(cl, noise, LMAX, N_CHAINS, seed)
    return [(shrink * draws[:N_DRAWS], truth) for draws, truth in pairs], cl, noise


def _top_z(chains, cl, noise):
    groups = dbe.edge_groups(LMAX)
    g = f"bin [{LMAX - 4},{LMAX})"
    obs = dbe.observed_u(chains, groups[g], thin=1)
    null = dbe.null_u(cl, noise, LMAX, {g: groups[g]}, 400, N_DRAWS, seed=5)[g]
    return (obs.mean() - null.mean()) / (null.std(ddof=1) / np.sqrt(obs.size))


def test_edge_groups_partition_the_top_bin():
    g = dbe.edge_groups(LMAX)
    top = g[f"bin [{LMAX - 4},{LMAX})"]
    per_l = sum(g[f"l = {L}"].astype(int) for L in range(LMAX - 4, LMAX))
    per_m = sum(g[k].astype(int) for k in ("top: m = 0", "top: 0 < m <= l/2", "top: m > l/2"))
    parts = g["top: real parts"].astype(int) + g["top: imag parts"].astype(int)
    for cover in (per_l, per_m, parts):
        assert np.array_equal(cover, top.astype(int))
    assert not (g[f"bin [{LMAX - 8},{LMAX - 4})"] & top).any()


def test_exact_sampler_reads_as_calibrated():
    chains, cl, noise = _ensemble()
    assert abs(_top_z(chains, cl, noise)) < 3.0
    L_arr, _ = nul.packed_layout(LMAX)
    sd, slope = dbe.z_and_slope(chains, L_arr, LMAX)
    top = slice(LMAX - 4, LMAX)
    # per-l sd(z) scatters by ~0.05-0.1 (every mode at l shares one C_l draw
    # per sky), so read it pooled over the bin, as the real result is read
    assert abs(sd[top].mean() - 1.0) < 0.08, sd[top]
    w = np.divide(cl, cl + noise, out=np.zeros_like(cl), where=cl > 0)
    assert np.all(np.abs(slope[top] - w[top]) < 0.1), (slope[top], w[top])


def test_shrunk_sampler_is_flagged_with_the_band_edge_sign():
    chains, cl, noise = _ensemble(shrink=0.8)
    assert _top_z(chains, cl, noise) > 3.0
    L_arr, _ = nul.packed_layout(LMAX)
    _, slope = dbe.z_and_slope(chains, L_arr, LMAX)
    w = np.divide(cl, cl + noise, out=np.zeros_like(cl), where=cl > 0)
    top = slice(LMAX - 4, LMAX)
    assert np.all(slope[top] < w[top] - 0.05)


def test_quarter_segments_cover_every_sweep():
    chains, _, _ = _ensemble()
    mask = dbe.edge_groups(LMAX)[f"bin [{LMAX - 4},{LMAX})"]
    full = dbe.observed_u(chains, mask, thin=1)
    quarters = [dbe.observed_u(chains, mask, 1, segment=(q, 4)) for q in range(4)]
    assert all(q.shape == full.shape for q in quarters)
    assert np.all((full > 0) & (full < 1))


def test_cl_ratio_by_quarter_is_flat_for_exact_draws_and_sees_a_low_start(tmp_path):
    chains, cl, noise = _ensemble()
    L_arr, v = nul.packed_layout(LMAX)
    from diffcmb.alm_utils import packed_dof_per_multipole

    k = packed_dof_per_multipole(LMAX)
    files = []
    for i, (draws, truth) in enumerate(chains):
        S = np.stack([np.bincount(L_arr, weights=a ** 2 / v, minlength=LMAX) for a in draws])
        rng = np.random.default_rng(i)
        # exact flat-prior C | a draws, the Block 1 conditional
        C = S[:, 2:] / 2.0 / rng.gamma(k[2:] / 2.0 - 1.0, size=S[:, 2:].shape)
        lnc = np.log(C)
        lnc[: draws.shape[0] // 4, -1] -= 1.0          # a low start at the top l
        f = tmp_path / f"chain_r{i:03d}.npz"
        np.savez(f, alm_samples=np.concatenate([lnc, draws], axis=1), alm_true_packed=truth)
        files.append(str(f))
    r = dbe.cl_ratio_by_quarter(files, LMAX)
    # exact draws: flat across quarters (the level itself exceeds 1 at this S/N)
    body = r[:, 2:LMAX - 1]
    assert np.all(np.abs(body - body.mean(axis=0)) < 0.12 * body.mean(axis=0))
    assert r[0, LMAX - 1] < 0.6 * r[3, LMAX - 1]
