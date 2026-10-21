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


def test_v1_qe_summary_is_the_trained_on_one(ctx):
    # The v1 flows were trained on this QE (C_tot over its whole range, the l = lmax
    # leg included). The width-gap refactor once band-limited it to l < lmax, which
    # moved ln C_L^QE by ~0.15 and biased every v1 flow scored afterwards (2026-10-15).
    import npe_simulate as ns

    from diffcmb import qe

    sim = ns.simulate(ctx, 5)
    phi = hp.almxfl(qe.qe_tt_reconstruct(sim["data_map"], ctx.cl_tt_lensed, ctx.cl_tot,
                                         LMAX, NSIDE), ctx.qe_norm)
    np.testing.assert_array_equal(sim["x"][LMAX - 2:], np.log(hp.alm2cl(phi, lmax=LMAX)[2:LMAX]))


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
    # a flow whose training summaries the current simulator no longer reproduces is refused
    orig = dict(np.load(sims / "train_0.npz"))
    blk = {**orig, "x": orig["x"].copy()}
    blk["x"][0, -1] += 0.1
    np.savez(sims / "train_0.npz", **blk)
    res = _cli("npe_vs_exact.py", "--flow", sims / "flow", "--chains_dir", strict,
               "--realizations", 1, "--n_learned", 200, "--out", tmp_path / "y.npz")
    assert res.returncode != 0 and "summaries" in res.stdout + res.stderr
    np.savez(sims / "train_0.npz", **orig)
    flat = tmp_path / "flat"
    _run_chain(flat, "--cl_phiphi_prior_nu", str(NU), realization=1)
    res = _cli("npe_vs_exact.py", "--flow", sims / "flow", "--chains_dir", flat,
               "--realizations", 1, "--out", tmp_path / "x.npz")
    assert res.returncode != 0 and "cl_prior_nu" in res.stdout + res.stderr


# --- summary set v2 (T1.1 width gap, route a) and the stored data a_lm (route b) ---

def _ctx(lmax, nside, noisesig, amp, summary_set):
    import npe_simulate as ns

    from diffcmb import qe
    from diffcmb.power import fiducial_spectra

    cl_tt, _ = fiducial_spectra(lmax + 1)
    cl_tt = cl_tt[: lmax + 1]
    nl = qe.qe_tt_noise_nl(cl_tt, cl_tt + qe.white_noise_cl(noisesig, 12 * nside ** 2, lmax), lmax)
    return ns.make_context(lmax=lmax, nside=nside, noisesig=noisesig, phi_amplitude=amp,
                           nu_tt=NU, nu_pp=NU, cl_tt_lensed=cl_tt, nl_qe=nl,
                           summary_set=summary_set)


@pytest.fixture(scope="module")
def ctx_v2():
    return _ctx(LMAX, NSIDE, 1.0, AMP, "v2")


def test_v2_summaries_extend_v1_on_the_same_sky(ctx, ctx_v2):
    import npe_simulate as ns

    n = LMAX - 2
    a, b = ns.simulate(ctx, 5), ns.simulate(ctx_v2, 5)
    assert b["x"].shape == (ns.summary_dim("v2", LMAX),) == (6 * n,)
    assert ns.summary_dim("v1", LMAX) == 2 * n
    np.testing.assert_array_equal(b["x"][:n], a["x"][:n])
    # v2's QE block is band-limited to 2 <= l < lmax (as the v2 sims were made), not v1's
    phi = ns.qe_phi(ctx_v2, b["data_map"], 2, LMAX)
    np.testing.assert_array_equal(b["x"][n: 2 * n], np.log(hp.alm2cl(phi, lmax=LMAX)[2:LMAX]))
    assert np.all(np.isfinite(b["x"]))
    corr = b["x"][4 * n: 5 * n]
    assert np.all(np.abs(corr) <= 1.0)
    np.testing.assert_array_equal(b["x"], ns.simulate(ctx_v2, 5)["x"])


def test_simulate_returns_the_data_alm(ctx):
    import npe_simulate as ns

    from diffcmb.alm_utils import packed_length
    from diffcmb.lensing import _alm_hp_to_packed

    s = ns.simulate(ctx, 5)
    assert s["data_alm_packed"].shape == (packed_length(LMAX),)
    np.testing.assert_allclose(
        s["data_alm_packed"],
        _alm_hp_to_packed(hp.map2alm(s["data_map"], lmax=LMAX - 1, iter=3), LMAX))


def test_delensing_by_the_true_phi_restores_the_unlensed_sky():
    """Pins the delensing sign: remapping the lensed map by -grad(phi_true) must
    raise its upper-band correlation with the unlensed sky, and +grad(phi) must
    lower it (exact-operator sky, noise-free, A_phi = 3000). Measured: 0.94
    lensed, 0.99 delensed, 0.78 wrong sign. Power alone is the wrong check: the
    remap of a band-limited field loses in-band power smoothly with l."""
    import npe_simulate as ns

    from diffcmb.lensing import _alm_packed_to_hp

    lmax = nside = 32
    c = _ctx(lmax, nside, 1e-6, AMP, "v2")
    s = ns.simulate(c, 7)
    lensed = s["data_map"]           # noise sigma 1e-6: the lensed sky
    phi = _alm_packed_to_hp(s["phi_true_packed"], lmax)
    unl = _alm_packed_to_hp(s["alm_true_packed"], lmax)

    def rho(m):
        a = hp.map2alm(m, lmax=lmax - 1, iter=3)
        return np.mean((hp.alm2cl(a, unl) / np.sqrt(hp.alm2cl(a) * hp.alm2cl(unl)))[lmax // 2:])

    r_len, r_del, r_wrong = rho(lensed), rho(ns.delens_map(c, lensed, phi)), rho(
        ns.delens_map(c, lensed, -phi))
    assert r_del > 0.98 and r_del > r_len + 0.03 and r_wrong < r_len - 0.05, (r_len, r_del, r_wrong)


def test_cli_v2_block_records_its_summary_set_and_data_alm(tmp_path):
    from diffcmb import qe
    from diffcmb.alm_utils import packed_length
    from diffcmb.power import fiducial_spectra

    cl_tt, _ = fiducial_spectra(LMAX + 1)
    cl_tt = cl_tt[: LMAX + 1]
    nl = qe.qe_tt_noise_nl(cl_tt, cl_tt + qe.white_noise_cl(1.0, 12 * NSIDE ** 2, LMAX), LMAX)
    np.savez(tmp_path / "qe.npz", cl_tt_lensed=cl_tt, nl_qe=nl, lmax=LMAX, nside=NSIDE,
             noisesig=1.0, phi_amplitude=AMP, lensing_operator="exact", fiducial="corrected",
             passed=True)
    res = _cli("npe_simulate.py", "--lmax", LMAX, "--nside", NSIDE, "--noisesig", 1,
               "--phi_amplitude", AMP, "--nu_tt", NU, "--nu_pp", NU, "--qe_file",
               tmp_path / "qe.npz", "--first", 100000, "--n", 2, "--summary_set", "v2",
               "--out", tmp_path / "b.npz")
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    b = np.load(tmp_path / "b.npz")
    assert str(b["summary_set"]) == "v2" and b["x"].shape == (2, 6 * (LMAX - 2))
    assert b["data_alm"].shape == (2, packed_length(LMAX)) and b["data_alm"].dtype == np.float32


def test_compare_sky_reports_the_width_ratio_per_multipole():
    # ROADMAP T1.1: the per-bin width ratio averages over modes; the per-l one says
    # whether the learned posterior is wider at every l or only at some.
    import npe_vs_exact as nve
    from npe_sbc import bin_slices

    lmax = 12
    rng = np.random.default_rng(0)
    exact = rng.standard_normal((4000, 2 * (lmax - 2)))
    scale = np.linspace(1.0, 2.0, 2 * (lmax - 2))
    learned = rng.standard_normal((4000, 2 * (lmax - 2))) * scale + 0.3
    out = nve.compare_sky(exact, learned, bin_slices(lmax))
    assert out["width_ratio_ell"].shape == (2 * (lmax - 2),)
    np.testing.assert_allclose(out["width_ratio_ell"], scale, rtol=0.06)
