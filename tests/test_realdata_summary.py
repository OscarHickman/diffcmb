"""scripts/realdata/realdata_summary.py: multi-chain R-hat, the one-data-set refusal and the Planck comparison."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import realdata_summary as rs  # noqa: E402

LMAX = 12


def _chain(path, rng, data, cl_tt, cl_pp_fid, shift=0.0, n=400, amp=3000.0):
    n_l = LMAX - 2
    ln_tt = np.log(cl_tt[2:]) + shift + 0.1 * rng.standard_normal((n, n_l))
    ln_pp = np.log(cl_pp_fid[2:] * 0.01) + 0.2 * rng.standard_normal((n, n_l))
    np.savez(path, lmax=LMAX, alm_samples=np.concatenate([ln_tt, rng.standard_normal((n, 5))], 1),
             cl_tt_fid=cl_tt,
             cl_phiphi_samples=ln_pp, cl_phiphi_fid=cl_pp_fid, phi_amplitude=amp,
             data_masked=data, data_map="smica.fits")


@pytest.fixture
def spectra():
    ell = np.arange(LMAX, dtype=float)
    cl_tt = np.r_[0, 0, 2 * np.pi * 1000.0 / (ell[2:] * (ell[2:] + 1))]
    return cl_tt, np.r_[0, 0, 3000.0 * 1e-7 / ell[2:] ** 4]


def test_chains_of_one_target_have_unit_rhat_and_a_shifted_chain_does_not(spectra):
    rng = np.random.default_rng(0)
    same = [rng.standard_normal((500, 3)) for _ in range(4)]
    assert np.all(rs.multichain_rhat(same) < 1.02)
    off = same[:3] + [same[3] + 1.0]
    assert np.all(rs.multichain_rhat(off) > 1.1)


def test_summary_reads_planck_and_the_phi_amplitude(tmp_path, spectra):
    cl_tt, cl_pp = spectra
    rng = np.random.default_rng(1)
    data = rng.standard_normal(30)
    for i in range(3):
        _chain(tmp_path / f"data_chain_r{i:03d}.npz", rng, data, cl_tt, cl_pp)
    files, ds = rs.load_chains(str(tmp_path))
    planck_dl = np.full(LMAX - 2, 1000.0)  # D_l of cl_tt
    s = rs.summarise(ds, planck_dl)
    np.testing.assert_allclose(s["C_l^TT/planck_over_median"], 1.0, atol=0.02)
    # posterior 0.01 x the prior centre, i.e. 30 x the physical fiducial at A_phi = 3000
    np.testing.assert_allclose(s["C_L^phiphi/over_prior_q50"], 0.01, rtol=0.05)
    np.testing.assert_allclose(s["C_L^phiphi/over_physical_q50"], 30.0, rtol=0.05)
    assert np.all(s["C_l^TT/rhat"] < 1.05)
    assert s["collapsed_ell"].size == 0


def test_a_collapsed_band_edge_multipole_is_named(tmp_path, spectra):
    cl_tt, cl_pp = spectra
    rng = np.random.default_rng(3)
    for i in range(2):
        _chain(tmp_path / f"data_chain_r{i:03d}.npz", rng, np.zeros(30), cl_tt, cl_pp)
        d = dict(np.load(tmp_path / f"data_chain_r{i:03d}.npz"))
        d["alm_samples"][:, LMAX - 3] -= 12.0  # l = LMAX - 1 stuck at the MAP clip
        np.savez(tmp_path / f"data_chain_r{i:03d}.npz", **d)
    s = rs.summarise(rs.load_chains(str(tmp_path))[1], np.full(LMAX - 2, 1000.0))
    np.testing.assert_array_equal(s["collapsed_ell"], [LMAX - 1])


def test_chains_that_saw_different_data_are_refused(tmp_path, spectra):
    rng = np.random.default_rng(2)
    _chain(tmp_path / "data_chain_r000.npz", rng, np.zeros(30), *spectra)
    _chain(tmp_path / "data_chain_r001.npz", rng, np.ones(30), *spectra)
    with pytest.raises(SystemExit, match="different data"):
        rs.load_chains(str(tmp_path))


def test_chains_whose_data_differ_by_roundoff_are_one_run(tmp_path, spectra):
    rng = np.random.default_rng(4)
    data = 60.0 * rng.standard_normal(30)
    _chain(tmp_path / "data_chain_r000.npz", rng, data, *spectra)
    _chain(tmp_path / "data_chain_r001.npz", rng, data + 6e-14, *spectra)  # an ulp, another CPU
    assert len(rs.load_chains(str(tmp_path))[1]) == 2


def test_planck_table_is_read_by_multipole(tmp_path):
    t = np.column_stack([np.arange(2, 40), 10.0 * np.arange(2, 40), np.ones(38), np.ones(38)])
    f = tmp_path / "tt.txt"
    np.savetxt(f, t)
    np.testing.assert_array_equal(rs.read_planck_dl(str(f), LMAX), 10.0 * np.arange(2, LMAX))
    with pytest.raises(SystemExit):
        rs.read_planck_dl(str(f), 64)


def _summary_file(tmp_path, name, spectra, amp, shift=0.0):
    cl_tt, cl_pp = spectra
    rng = np.random.default_rng(int(amp) % 97)
    d = tmp_path / name
    d.mkdir()
    for i in range(2):
        _chain(d / f"data_chain_r{i:03d}.npz", rng, np.zeros(30), cl_tt, cl_pp, shift=shift, amp=amp)
    planck = (np.full(LMAX - 2, 1000.0), np.full(LMAX - 2, 50.0), np.full(LMAX - 2, 50.0))
    s = rs.summarise(rs.load_chains(str(d))[1], planck[0])
    s["planck_dl_ell"], s["planck_dl_lo"], s["planck_dl_hi"] = planck
    np.savez(d / "summary.npz", **{k: np.asarray(v) for k, v in s.items()})
    return d / "summary.npz"


def test_realdata_figure_builds_and_refuses_a_collapsed_run(tmp_path, spectra):
    import subprocess
    repo = os.path.join(os.path.dirname(__file__), "..")
    fig = os.path.join(repo, "scripts", "paper", "fig_realdata.py")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [os.path.join(repo, d) for d in ("diffcmb", "scripts", "scripts/paper")]))
    a1 = _summary_file(tmp_path, "a1", spectra, 1.0)
    a3 = _summary_file(tmp_path, "a3", spectra, 3000.0)
    res = subprocess.run([sys.executable, fig, "--a1", str(a1), "--a3000", str(a3), "--sevem", str(a1),
                          "--outdir", str(tmp_path / "out")], env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]
    assert (tmp_path / "out" / "realdata_cltt.pdf").exists() and (tmp_path / "out" / "realdata_clpp.pdf").exists()
    bad = _summary_file(tmp_path, "bad", spectra, 1.0, shift=-12.0)
    res = subprocess.run([sys.executable, fig, "--a1", str(bad), "--a3000", str(a3),
                          "--outdir", str(tmp_path / "out2")], env=env, capture_output=True, text=True)
    assert res.returncode != 0 and "collapsed" in res.stdout + res.stderr


def test_map_comparison_reports_a_shift_in_posterior_sd_and_refuses_unlike_runs(tmp_path, spectra):
    import realdata_compare as rc
    a = np.load(_summary_file(tmp_path, "a", spectra, 1.0))
    b = np.load(_summary_file(tmp_path, "b", spectra, 1.0, shift=0.05))
    c = rc.compare(a, b)
    np.testing.assert_allclose(c["ell/ratio"], np.exp(0.05), rtol=0.03)
    np.testing.assert_allclose(c["ell/shift_sd"], 0.5, atol=0.25)  # 0.05 against a 0.1 sd
    assert np.all(rc.compare(a, a)["bin/shift_sd"] == 0)
    with pytest.raises(SystemExit, match="phi_amplitude"):
        rc.compare(a, np.load(_summary_file(tmp_path, "c", spectra, 3000.0)))


def test_realdata_figure_refuses_chains_that_disagree(tmp_path, spectra):
    import subprocess
    repo = os.path.join(os.path.dirname(__file__), "..")
    fig = os.path.join(repo, "scripts", "paper", "fig_realdata.py")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [os.path.join(repo, d) for d in ("diffcmb", "scripts", "scripts/paper")]))
    a3 = _summary_file(tmp_path, "a3", spectra, 3000.0)
    s = dict(np.load(_summary_file(tmp_path, "a1", spectra, 1.0)))
    s["C_L^phiphi/rhat"] = s["C_L^phiphi/rhat"] + 0.5  # one chain ran off
    np.savez(tmp_path / "split.npz", **s)
    res = subprocess.run([sys.executable, fig, "--a1", str(tmp_path / "split.npz"), "--a3000", str(a3),
                          "--outdir", str(tmp_path / "out")], env=env, capture_output=True, text=True)
    assert res.returncode != 0 and "R-hat" in res.stdout + res.stderr


def test_realdata_cltt_panel_builds_while_clpp_has_not_converged(tmp_path, spectra):
    import subprocess
    repo = os.path.join(os.path.dirname(__file__), "..")
    fig = os.path.join(repo, "scripts", "paper", "fig_realdata.py")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [os.path.join(repo, d) for d in ("diffcmb", "scripts", "scripts/paper")]))
    s = dict(np.load(_summary_file(tmp_path, "a1", spectra, 1.0)))
    s["C_L^phiphi/rhat"] = s["C_L^phiphi/rhat"] + 0.5  # the A_phi = 1 phi trap
    np.savez(tmp_path / "trap.npz", **s)
    run = lambda *a: subprocess.run([sys.executable, fig, "--a1", str(tmp_path / "trap.npz"), *a],  # noqa: E731
                                    env=env, capture_output=True, text=True)
    res = run("--panels", "cltt", "--outdir", str(tmp_path / "out"))
    assert res.returncode == 0, res.stderr[-2000:]
    assert (tmp_path / "out" / "realdata_cltt.pdf").exists()
    assert not (tmp_path / "out" / "realdata_clpp.pdf").exists()
    # the C_l^TT check still applies, to the second map as well
    t = dict(s)
    t["C_l^TT/rhat"] = t["C_l^TT/rhat"] + 0.5
    np.savez(tmp_path / "sevem_split.npz", **t)
    res = run("--panels", "cltt", "--sevem", str(tmp_path / "sevem_split.npz"),
              "--outdir", str(tmp_path / "out2"))
    assert res.returncode != 0 and "--sevem: C_l^TT R-hat" in res.stdout + res.stderr
    # the clpp panel still refuses the trapped run
    a3 = _summary_file(tmp_path, "a3", spectra, 3000.0)
    res = run("--a3000", str(a3), "--outdir", str(tmp_path / "out3"))
    assert res.returncode != 0 and "C_L^phiphi R-hat" in res.stdout + res.stderr


def _draw(path, rng, data, spectra, shift, n_chains=1, n=400):
    path.mkdir()
    for i in range(n_chains):
        _chain(path / f"data_chain_r{i:03d}.npz", rng, data, *spectra, shift=shift, n=n, amp=1.0)
    return str(path)


def test_noise_draws_pool_with_equal_weight_and_report_their_spread(tmp_path, spectra):
    rng = np.random.default_rng(5)
    # draw 0 has 4 chains, draw 1 one shorter chain: each must count once
    a = _draw(tmp_path / "nr0", rng, rng.standard_normal(30), spectra, -0.2, n_chains=4)
    b = _draw(tmp_path / "nr1", rng, rng.standard_normal(30), spectra, +0.2, n=300)
    files, draws = rs.load_draws([a, b])
    assert len(files) == 5 and [len(d) for d in draws] == [4, 1]
    s = rs.summarise_draws(draws, np.full(LMAX - 2, 1000.0))
    assert int(s["n_draws"]) == 2 and int(s["n_chains"]) == 5
    # equal weight: the pooled median sits between the draws, not on the 4-chain one
    np.testing.assert_allclose(np.log(s["C_l^TT/planck_over_median"]), 0.0, atol=0.03)
    # between-draw sd 0.2 against a per-bin posterior sd of 0.1 / sqrt(bin width)
    assert np.all(s["C_l^TT/draw_spread_sd"][:-1] > 2.0)
    # per-draw R-hat: two draws that disagree are not a convergence failure
    assert np.all(s["C_l^TT/rhat"] < 1.05)


def test_one_run_has_no_draw_spread(tmp_path, spectra):
    rng = np.random.default_rng(6)
    a = _draw(tmp_path / "nr0", rng, rng.standard_normal(30), spectra, 0.0, n_chains=2)
    s = rs.summarise(rs.load_chains(a)[1], np.full(LMAX - 2, 1000.0))
    assert int(s["n_draws"]) == 1 and "C_l^TT/draw_spread_sd" not in s


def test_draws_of_different_maps_or_the_same_noise_are_refused(tmp_path, spectra):
    rng = np.random.default_rng(7)
    data = rng.standard_normal(30)
    a = _draw(tmp_path / "nr0", rng, data, spectra, 0.0)
    b = _draw(tmp_path / "nr0_again", rng, data, spectra, 0.0)
    with pytest.raises(SystemExit, match="same data"):
        rs.load_draws([a, b])
    c = _draw(tmp_path / "sevem", rng, rng.standard_normal(30), spectra, 0.0)
    d = dict(np.load(tmp_path / "sevem" / "data_chain_r000.npz"))
    d["data_map"] = "sevem.fits"
    np.savez(tmp_path / "sevem" / "data_chain_r000.npz", **d)
    with pytest.raises(SystemExit, match="map"):
        rs.load_draws([a, c])
