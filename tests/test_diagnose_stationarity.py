"""scripts/diagnose_calibration_stationarity.py (ROADMAP T0.1 a, b).

The script's job is to tell burn-in drift from a stationary offset, so each
test builds a synthetic ensemble with a known answer: exchangeable draws (a
correct, stationary sampler), draws that relax toward the posterior along the
chain (burn-in too short), and draws with a constant offset (a biased but
stationary sampler).
"""

from __future__ import annotations

import os
import re
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

import diagnose_calibration_stationarity as dcs  # noqa: E402

from diffcmb.alm_utils import packed_length  # noqa: E402

LMAX = 12


def _ensemble(dirpath, n_chains=24, n_samp=400, drift=0.0, offset=0.0, seed=3):
    """Draws are mu + offset + drift*exp(-t/100) + N(0,1); truth is mu + N(0,1).

    drift != 0 puts a transient at the start of the saved chain; offset != 0
    is a stationary bias.
    """
    rng = np.random.default_rng(seed)
    n_pk = packed_length(LMAX)
    t = np.arange(n_samp)[:, None]
    for c in range(n_chains):
        fields = {}
        for name in ("phi", "alm"):
            mu = rng.normal(size=n_pk)
            fields[name + "_true"] = mu + rng.normal(size=n_pk)
            fields[name] = (mu + offset + drift * np.exp(-t / 100.0)
                            + rng.normal(size=(n_samp, n_pk)))
        np.savez(os.path.join(dirpath, f"chain_r{c:03d}.npz"), lmax=LMAX,
                 alm_samples=np.hstack([np.zeros((n_samp, LMAX - 2)), fields["alm"]]),
                 phi_samples=fields["phi"], alm_true_packed=fields["alm_true"],
                 phi_true_packed=fields["phi_true"], logp=rng.normal(size=n_samp))


def _traces(tmp_path, **kw):
    _ensemble(tmp_path, **kw)
    return dcs.load_traces(dcs.chain_files(str(tmp_path)))


def test_trace_rank_equals_rank_of_power_against_truth(tmp_path):
    traces = _traces(tmp_path)
    table = dcs.rank_table(traces, thin=1, window="full")
    ranks, n_draws = table[("alm_power", 2, 10)]
    assert n_draws == 399
    assert ranks.min() >= 0 and ranks.max() <= 400


def test_stationary_correct_sampler_shows_no_drift(tmp_path):
    traces = _traces(tmp_path)
    for key in [("alm_power", 2, 10), ("phi_power", 10, 12), "logp"]:
        assert abs(dcs.drift_z(dcs.segment_level(traces, key, 4))) < 3.5
    seg = dcs.segment_mean_u(traces, 4)[("alm_power", 2, 10)].mean(axis=0)
    assert np.all(np.abs(seg - 0.5) < 0.2)


def test_burn_in_transient_is_flagged_as_drift(tmp_path):
    traces = _traces(tmp_path, drift=1.5)
    key = ("alm_power", 2, 10)
    assert dcs.drift_z(dcs.segment_level(traces, key, 4)) < -5
    seg = dcs.segment_mean_u(traces, 4)[key].mean(axis=0)
    assert seg[0] < seg[-1] - 0.1          # truth ranks low early, recovers late


def test_stationary_offset_is_flat_across_segments_but_off_centre(tmp_path):
    traces = _traces(tmp_path, offset=1.0)
    key = ("alm_power", 2, 10)
    assert abs(dcs.drift_z(dcs.segment_level(traces, key, 4))) < 3.5
    seg = dcs.segment_mean_u(traces, 4)[key].mean(axis=0)
    assert np.all(seg < 0.4) and np.ptp(seg) < 0.1


def test_report_runs_end_to_end(tmp_path, capsys):
    _ensemble(tmp_path, n_chains=6, n_samp=120)
    dcs.report(str(tmp_path), thins=[1, 10], n_seg=4)
    out = capsys.readouterr().out
    assert "thin=10 window=second" in out
    assert re.search(r"logp\s+rel\. to seg 0", out)
