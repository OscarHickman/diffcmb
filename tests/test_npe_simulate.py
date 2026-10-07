"""npe_simulate (ROADMAP T1.1): the learned posterior's training simulator.

The learned-vs-exact comparison is only meaningful if the NPE is trained on the
SAME generative process the exact strict ensemble's truths come from, so the
simulator is pinned to coverage_ensemble_chain's strict mode bit for bit.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

hp = pytest.importorskip("healpy")
pytest.importorskip("tensorflow")
pytest.importorskip("camb")

from test_scripts_pipeline import LMAX, NSIDE, REPO, SCRIPTS, _run_chain  # noqa: E402

NU = 30.0
AMP = 3000.0


@pytest.fixture(scope="module")
def ctx():
    import npe_simulate as ns

    from diffcmb import qe
    from diffcmb.power import fiducial_spectra

    cl_tt, _ = fiducial_spectra(LMAX + 1)
    cl_tt = cl_tt[: LMAX + 1]
    cl_tot = cl_tt + qe.white_noise_cl(1.0, 12 * NSIDE ** 2, LMAX)
    nl = qe.qe_tt_noise_nl(cl_tt, cl_tot, LMAX)
    return ns.make_context(lmax=LMAX, nside=NSIDE, noisesig=1.0, phi_amplitude=AMP,
                           nu_tt=NU, nu_pp=NU, cl_tt_lensed=cl_tt, nl_qe=nl)


def test_truth_matches_the_strict_chain_bit_for_bit(tmp_path, ctx):
    import npe_simulate as ns

    _run_chain(tmp_path, "--cl_prior_nu", str(NU), "--cl_phiphi_prior_nu", str(NU),
               realization=3)
    d = np.load(tmp_path / "chain_r003.npz")
    sim = ns.simulate(ctx, 3)
    for key in ("cl_true", "cl_phiphi_true", "alm_true_packed", "phi_true_packed"):
        np.testing.assert_array_equal(sim[key], d[key], err_msg=key)
    np.testing.assert_array_equal(
        sim["theta"], np.log(np.concatenate([d["cl_true"][2:LMAX],
                                             d["cl_phiphi_true"][2:LMAX]])))


def test_summaries_are_finite_deterministic_and_sky_dependent(ctx):
    import npe_simulate as ns

    a, b, c = ns.simulate(ctx, 5), ns.simulate(ctx, 5), ns.simulate(ctx, 6)
    assert a["x"].shape == (2 * (LMAX - 2),) and np.all(np.isfinite(a["x"]))
    np.testing.assert_array_equal(a["x"], b["x"])
    assert not np.allclose(a["x"], c["x"])
    # first half is the data power spectrum of the simulated map
    np.testing.assert_allclose(
        a["x"][: LMAX - 2], np.log(hp.anafast(a["data_map"], lmax=LMAX - 1)[2:LMAX]),
        rtol=1e-12)


def test_simulate_refuses_a_realization_outside_its_stream_block(ctx):
    import npe_simulate as ns

    with pytest.raises(ValueError):
        ns.simulate(ctx, 1_000_000)


def test_cli_writes_a_training_block_and_refuses_a_mismatched_qe_file(tmp_path):
    from diffcmb import qe
    from diffcmb.power import fiducial_spectra

    cl_tt, _ = fiducial_spectra(LMAX + 1)
    cl_tt = cl_tt[: LMAX + 1]
    nl = qe.qe_tt_noise_nl(cl_tt, cl_tt + qe.white_noise_cl(1.0, 12 * NSIDE ** 2, LMAX), LMAX)
    good = {"cl_tt_lensed": cl_tt, "nl_qe": nl, "lmax": LMAX, "nside": NSIDE, "noisesig": 1.0,
            "phi_amplitude": AMP, "lensing_operator": "exact", "fiducial": "corrected",
            "passed": True}
    np.savez(tmp_path / "qe.npz", **good)
    np.savez(tmp_path / "qe_bad.npz", **{**good, "phi_amplitude": 1.0})
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "diffcmb"), OMP_NUM_THREADS="2")
    base = [sys.executable, os.path.join(SCRIPTS, "npe_simulate.py"), "--lmax", str(LMAX),
            "--nside", str(NSIDE), "--noisesig", "1", "--phi_amplitude", str(AMP),
            "--nu_tt", str(NU), "--nu_pp", str(NU), "--first", "100000", "--n", "3"]
    ok = subprocess.run(base + ["--qe_file", str(tmp_path / "qe.npz"),
                                "--out", str(tmp_path / "block.npz")],
                        env=env, capture_output=True, text=True, timeout=600)
    assert ok.returncode == 0, ok.stdout[-2000:] + ok.stderr[-2000:]
    blk = np.load(tmp_path / "block.npz")
    assert blk["theta"].shape == (3, 2 * (LMAX - 2)) and blk["x"].shape == (3, 2 * (LMAX - 2))
    np.testing.assert_array_equal(blk["realization"], [100000, 100001, 100002])
    assert float(blk["nu_tt"]) == NU and float(blk["phi_amplitude"]) == AMP
    bad = subprocess.run(base + ["--qe_file", str(tmp_path / "qe_bad.npz"),
                                 "--out", str(tmp_path / "x.npz")],
                         env=env, capture_output=True, text=True, timeout=600)
    assert bad.returncode != 0 and "phi_amplitude" in bad.stderr + bad.stdout


def _cli(script, *args):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(REPO, "diffcmb"), SCRIPTS]),
               OMP_NUM_THREADS="2")
    return subprocess.run([sys.executable, os.path.join(SCRIPTS, script), *map(str, args)],
                          env=env, capture_output=True, text=True, timeout=1200)


def test_npe_vs_exact_runs_on_strict_chains_and_refuses_others(tmp_path):
    # ROADMAP T1.1 figures 2-3: the comparison rebuilds each sky from the
    # simulator, checks it against the chain's truth, and refuses a chain run
    # under a different prior.
    from diffcmb import qe
    from diffcmb.power import fiducial_spectra

    cl_tt, _ = fiducial_spectra(LMAX + 1)
    cl_tt = cl_tt[: LMAX + 1]
    nl = qe.qe_tt_noise_nl(cl_tt, cl_tt + qe.white_noise_cl(1.0, 12 * NSIDE ** 2, LMAX), LMAX)
    qe_file = tmp_path / "qe.npz"
    np.savez(qe_file, cl_tt_lensed=cl_tt, nl_qe=nl, lmax=LMAX, nside=NSIDE, noisesig=1.0,
             phi_amplitude=AMP, lensing_operator="exact", fiducial="corrected", passed=True)
    sims = tmp_path / "sims"
    res = _cli("npe_simulate.py", "--lmax", LMAX, "--nside", NSIDE, "--noisesig", 1,
               "--phi_amplitude", AMP, "--nu_tt", NU, "--nu_pp", NU, "--qe_file", qe_file,
               "--first", 100000, "--n", 1000, "--out", sims / "train_0.npz")
    assert res.returncode == 0, res.stderr[-2000:]
    res = _cli("npe_train.py", "--sims_dir", sims, "--out", sims / "flow", "--n_layers", 2,
               "--hidden", 32, 32, "--epochs", 5)
    assert res.returncode == 0, res.stderr[-2000:]
    strict = tmp_path / "strict"
    for r in (1, 2):
        _run_chain(strict, "--cl_prior_nu", str(NU), "--cl_phiphi_prior_nu", str(NU), realization=r)
    res = _cli("npe_vs_exact.py", "--flow", sims / "flow", "--chains_dir", strict,
               "--realizations", 1, 2, "--n_learned", 200, "--out", tmp_path / "cmp.npz")
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    c = np.load(tmp_path / "cmp.npz")
    assert c["shift"].shape[0] == 2 and c["shift"].shape == c["width_ratio"].shape
    assert np.all(np.isfinite(c["shift"])) and np.all(c["width_ratio"] > 0)
    # exact draws are on the truth's (log) scale for both spectra
    import npe_vs_exact as nve

    d = np.load(strict / "chain_r001.npz")
    th = nve.exact_theta(d, LMAX)
    truth = np.log(np.concatenate([d["cl_true"][2:LMAX], d["cl_phiphi_true"][2:LMAX]]))
    assert np.all(np.abs(th.mean(0) - truth) < 2.0)
    flat = tmp_path / "flat"
    _run_chain(flat, "--cl_phiphi_prior_nu", str(NU), realization=1)
    res = _cli("npe_vs_exact.py", "--flow", sims / "flow", "--chains_dir", flat,
               "--realizations", 1, "--out", tmp_path / "x.npz")
    assert res.returncode != 0 and "cl_prior_nu" in res.stdout + res.stderr
