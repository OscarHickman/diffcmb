"""npe_train -> npe_sbc end to end (ROADMAP T1.1), on linear-Gaussian blocks in
npe_simulate's file format: a calibrated flow passes the SBC, a corrupted
held-out set is flagged, and mismatched or overlapping sims are refused."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("tensorflow_probability")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")
LMAX = 6
DIM = 2 * (LMAX - 2)
CFG = {"lmax": LMAX, "nside": 8, "noisesig": 1.0, "phi_amplitude": 3000.0, "nu_tt": 30.0,
       "nu_pp": 30.0, "lensing_operator": "exact", "fiducial": "corrected", "qe_file": "qe.npz"}


def _block(path, first, n, seed, x_scale=1.0, **over):
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(n, DIM))
    x = x_scale * (theta + 0.5 * rng.normal(size=(n, DIM)))
    np.savez(path, theta=theta, x=x, realization=np.arange(first, first + n), **{**CFG, **over})


def _run(script, *args):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(REPO, "diffcmb"), SCRIPTS]),
               OMP_NUM_THREADS="2")
    return subprocess.run([sys.executable, os.path.join(SCRIPTS, script), *map(str, args)],
                          env=env, capture_output=True, text=True, timeout=900)


@pytest.fixture(scope="module")
def flow_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("npe")
    _block(d / "train_0.npz", 100000, 8000, 0)
    _block(d / "train_1.npz", 108000, 8000, 1)
    res = _run("npe_train.py", "--sims_dir", d, "--out", d / "flow", "--n_layers", 2,
               "--hidden", 32, 32, "--epochs", 15, "--batch_size", 256, "--learning_rate", 2e-3)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    return d


def test_calibrated_flow_passes_sbc(flow_dir):
    _block(flow_dir / "heldout.npz", 500000, 400, 2)
    res = _run("npe_sbc.py", "--flow", flow_dir / "flow", "--heldout", flow_dir / "heldout.npz",
               "--n_sims", 400, "--n_draws", 39, "--out", flow_dir / "sbc.npz")
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    s = np.load(flow_dir / "sbc.npz")
    assert s["component_ranks"].shape == (400, DIM)
    assert s["bin_ranks"].shape == (400, 2)  # one bin [2,6) per spectrum at lmax 6
    assert np.all(s["bin_p"] > 0.001), s["bin_p"]


def test_sbc_flags_a_corrupted_heldout_set(flow_dir):
    _block(flow_dir / "heldout_bad.npz", 500000, 400, 3, x_scale=1.5)
    res = _run("npe_sbc.py", "--flow", flow_dir / "flow", "--heldout", flow_dir / "heldout_bad.npz",
               "--n_sims", 400, "--n_draws", 39, "--out", flow_dir / "sbc_bad.npz")
    assert res.returncode == 0, res.stderr[-2000:]
    assert np.any(np.load(flow_dir / "sbc_bad.npz")["bin_p"] < 0.001)


def test_sbc_refuses_mismatched_or_overlapping_sims(flow_dir):
    _block(flow_dir / "h_cfg.npz", 500000, 50, 4, phi_amplitude=1.0)
    _block(flow_dir / "h_overlap.npz", 100010, 50, 5)
    for name, msg in (("h_cfg.npz", "phi_amplitude"), ("h_overlap.npz", "overlap")):
        res = _run("npe_sbc.py", "--flow", flow_dir / "flow", "--heldout", flow_dir / name,
                   "--n_sims", 50, "--out", flow_dir / "x.npz")
        assert res.returncode != 0 and msg in res.stdout + res.stderr


def test_train_refuses_blocks_with_different_configurations(tmp_path):
    _block(tmp_path / "train_0.npz", 100000, 100, 0)
    _block(tmp_path / "train_1.npz", 100100, 100, 1, noisesig=30.0)
    res = _run("npe_train.py", "--sims_dir", tmp_path, "--out", tmp_path / "flow", "--epochs", 1)
    assert res.returncode != 0 and "noisesig" in res.stdout + res.stderr
