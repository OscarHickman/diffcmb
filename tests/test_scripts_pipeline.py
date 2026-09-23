"""Tests for the production pipeline scripts (scripts/), end to end at toy size.

The package tests (test_lensing, test_samplers, ...) cover diffcmb/; nothing
covered the scripts that generate, run and re-analyse the production chains,
and that is where silent failures have landed: an edit that put chain metadata
into the Gibbs call instead of the save call (2026-09-23), replays that would
have rebuilt a chain's data with the wrong lensing operator, and the rank
statistics every figure prints.

  * coverage_ensemble_chain  -- aware and blind modes run end to end, share one
                                sky, record operator / fiducial / amplitude /
                                noise, and apply the amplitude to the prior
  * sbc_joint_likelihood     -- replay reproduces the saved truth and uses the
                                chain's own operator; legacy chains default to
                                the bilinear operator they were made with
  * aggregate_coverage_ranks -- rank helpers and the calibrated discrete test
  * validate_sbc_statistic_power -- the power model behind the quoted bound
  * power.fiducial_spectra   -- the corrected fiducial
  * model / psi_lensed       -- fp64 prior, prior on the un-beamed alm
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

hp = pytest.importorskip("healpy")
tf = pytest.importorskip("tensorflow")
pytest.importorskip("camb")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from diffcmb.alm_utils import packed_dof_per_multipole, packed_length  # noqa: E402

LMAX, NSIDE, AMP = 8, 8, 3000.0


def _run_chain(outdir, *extra, realization=1):
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "diffcmb"), OMP_NUM_THREADS="2")
    cmd = [sys.executable, os.path.join(SCRIPTS, "coverage_ensemble_chain.py"),
           "--realization", str(realization), "--lmax", str(LMAX), "--nside", str(NSIDE),
           "--n_burnin", "3", "--n_samples", "6", "--map_steps", "5",
           "--n_lfs", "3", "--phi_n_lfs", "5", "--cl_phiphi_prior_nu", "6",
           "--phi_amplitude", str(AMP), "--lensing_operator", "exact",
           "--checkpoint_every", "100", "--outdir", str(outdir), *extra]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    return res.stdout


@pytest.fixture(scope="module")
def smoke_ensemble(tmp_path_factory):
    out = tmp_path_factory.mktemp("ens")
    for r in (1, 2):
        _run_chain(out, realization=r)
        _run_chain(out, "--blind", realization=r)
    return out


# ---------------------------------------------------------------------------
# coverage_ensemble_chain
# ---------------------------------------------------------------------------

def test_aware_chain_records_its_configuration(smoke_ensemble):
    d = np.load(smoke_ensemble / "chain_r001.npz")
    assert str(d["lensing_operator"]) == "exact"
    assert str(d["fiducial"]) == "corrected"
    assert float(d["phi_amplitude"]) == AMP
    assert float(d["noisesig"]) == 1.0
    assert int(d["packing_version"]) == 2
    assert d["phi_samples"].shape == (6, packed_length(LMAX))
    assert d["alm_samples"].shape == (6, LMAX - 2 + packed_length(LMAX))
    assert d["cl_phiphi_samples"].shape == (6, LMAX - 2)
    assert np.all(np.isfinite(d["alm_samples"])) and np.all(np.isfinite(d["phi_samples"]))


def test_amplitude_scales_the_prior_fiducial(smoke_ensemble):
    from diffcmb.power import fiducial_spectra

    d = np.load(smoke_ensemble / "chain_r001.npz")
    _, cl_pp = fiducial_spectra(LMAX)
    np.testing.assert_allclose(d["cl_phiphi_fid"][2:], AMP * cl_pp[2:], rtol=1e-6)


def test_blind_chain_fits_the_same_sky_without_a_phi_block(smoke_ensemble):
    a = np.load(smoke_ensemble / "chain_r001.npz")
    b = np.load(smoke_ensemble / "blind_r001.npz")
    assert bool(b["blind"])
    assert "phi_samples" not in b.files
    np.testing.assert_array_equal(a["alm_true_packed"], b["alm_true_packed"])
    np.testing.assert_array_equal(a["phi_true_packed"], b["phi_true_packed"])
    np.testing.assert_array_equal(a["cl_true"], b["cl_true"])
    assert str(b["lensing_operator"]) == "exact"


def test_chain_script_rejects_a_nonpositive_amplitude(tmp_path):
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "diffcmb"))
    res = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS, "coverage_ensemble_chain.py"),
         "--realization", "0", "--phi_amplitude", "0", "--outdir", str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=300)
    assert res.returncode != 0 and "phi_amplitude" in (res.stdout + res.stderr)


# ---------------------------------------------------------------------------
# sbc_joint_likelihood replay
# ---------------------------------------------------------------------------

def test_replay_rebuilds_the_saved_truth_with_the_chains_operator(smoke_ensemble):
    import sbc_joint_likelihood as sjl

    from diffcmb import CosmologyAdvancedSampling

    d = np.load(smoke_ensemble / "chain_r001.npz")
    assert sjl.chain_lensing_operator(d) == "exact"
    assert sjl.chain_noisesig(d, 99.0) == 1.0
    model = CosmologyAdvancedSampling(
        _lmax=LMAX, _NSIDE=NSIDE, _noisesig=1.0, data_mode="synthetic",
        dtype=tf.complex128, use_matrixfree_sht=True, lensing_operator="exact")
    model._ensure_tf_tensors()
    alm_t, phi_t, y = sjl.rebuild_truth_and_data(model, d, LMAX, 1.0)
    np.testing.assert_allclose(alm_t, d["alm_true_packed"], atol=1e-12)
    np.testing.assert_allclose(phi_t, d["phi_true_packed"], atol=1e-12)
    assert y.shape == (hp.nside2npix(NSIDE),)


def test_legacy_chain_without_metadata_replays_with_bilinear(tmp_path):
    import sbc_joint_likelihood as sjl

    np.savez(tmp_path / "old.npz", lmax=8)
    d = np.load(tmp_path / "old.npz")
    assert sjl.chain_lensing_operator(d) == "bilinear"
    assert sjl.chain_noisesig(d, 1.0) == 1.0


# ---------------------------------------------------------------------------
# rank helpers
# ---------------------------------------------------------------------------

def test_rank_of_and_binned_power():
    import aggregate_coverage_ranks as acr

    assert acr.rank_of(0.5, [0.1, 0.2, 0.9]) == 2
    assert acr.rank_of(-1.0, [0.1, 0.2]) == 0
    coeffs = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 1.0]])
    L = np.array([2, 2, 3])
    np.testing.assert_allclose(acr.binned_power(coeffs, L, 2, 3), [2.5, 0.5])
    assert acr.binned_power(coeffs, L, 10, 20) is None


def test_discrete_uniform_p_is_calibrated_and_powerful():
    import aggregate_coverage_ranks as acr

    rng = np.random.default_rng(0)
    ps = [acr.discrete_uniform_p(rng.integers(0, 8, size=48), 7, n_rep=2000,
                                 seed=int(rng.integers(1e9))) for _ in range(150)]
    assert 0.01 <= np.mean(np.array(ps) < 0.05) <= 0.11
    assert acr.discrete_uniform_p(np.zeros(48, dtype=int), 7, n_rep=2000) < 0.001


def test_rank_spread_of_uniform_is_one_over_root_twelve():
    import aggregate_coverage_ranks as acr

    ranks = np.repeat(np.arange(60), 50)
    assert acr.rank_spread(ranks, 59) == pytest.approx(1 / np.sqrt(12), rel=0.01)


def test_realized_spectrum_divides_by_the_packed_dof():
    import aggregate_coverage_ranks as acr

    lmax = 10
    S = np.arange(lmax, dtype=float) * 3.0
    cl = acr.realized_spectrum(S, lmax)
    k = packed_dof_per_multipole(lmax)
    np.testing.assert_allclose(cl[2:], S[2:] / k[2:])
    np.testing.assert_array_equal(k[2:], 2 * np.arange(2, lmax) + 1)


def test_sbc_power_model_is_uniform_without_a_shift_and_moves_with_one():
    import validate_sbc_statistic_power as vp

    rng = np.random.default_rng(1)
    u0 = np.concatenate([vp.sbc_ranks(200, 60, rng) for _ in range(5)])
    assert abs(u0.mean() - 0.5) < 0.03
    u1 = np.concatenate([vp.sbc_ranks(200, 60, rng, mean_shift=1.0) for _ in range(5)])
    assert abs(u1.mean() - 0.5) > 0.15


# ---------------------------------------------------------------------------
# corrected fiducial
# ---------------------------------------------------------------------------

def test_fiducial_uses_physical_densities():
    from diffcmb.power import LCDM_PARAMS, LCDM_PARAMS_LEGACY

    h = LCDM_PARAMS[0] / 100.0
    assert LCDM_PARAMS[1] == pytest.approx(LCDM_PARAMS_LEGACY[1] * h ** 2)
    assert 0.020 < LCDM_PARAMS[1] < 0.025          # omega_b
    assert 0.10 < LCDM_PARAMS[2] < 0.14            # omega_c


def test_fiducial_spectra_are_raw_unlensed_cls():
    from diffcmb.power import fiducial_spectra

    tt, pp = fiducial_spectra(200)
    ell = np.arange(200)
    dl = ell * (ell + 1) * tt / (2 * np.pi)
    assert 600 < dl[10] < 1300                     # Sachs-Wolfe plateau, muK^2
    assert 4000 < dl[190:200].max() < 7000         # rising to the first peak
    lpp = (ell * (ell + 1.0)) ** 2 * pp / (2 * np.pi)
    assert 0.5e-7 < lpp[40] < 2e-7                 # standard [L(L+1)]^2 C/2pi
    assert tt[0] == tt[1] == pp[0] == pp[1] == 0.0


# ---------------------------------------------------------------------------
# model precision and the beam/prior ordering
# ---------------------------------------------------------------------------

def test_prior_weights_are_float64():
    from diffcmb import CosmologyAdvancedSampling

    m = CosmologyAdvancedSampling(_lmax=8, _NSIDE=8, _noisesig=1.0, data_mode="synthetic",
                                  dtype=tf.complex128, use_matrixfree_sht=True)
    m._ensure_tf_tensors()
    assert m.l_weights.dtype == tf.float64


def test_psi_lensed_prior_is_on_the_unbeamed_alm():
    """With a beam, the prior must use the sky alm; only the forward model
    sees the beam. Checked by differencing psi_lensed between two noise levels:
    the prior part is noise-independent, the likelihood part scales as 1/N."""
    from diffcmb import CosmologyAdvancedSampling
    from diffcmb.alm_utils import splittosingularalm
    from diffcmb.lensing import psi_lensed

    lmax, nside = 8, 8
    rng = np.random.default_rng(3)
    params = np.concatenate([np.full(lmax - 2, 1.0),
                             rng.standard_normal(packed_length(lmax))])
    phi = tf.constant(np.zeros(packed_length(lmax)), tf.float64)
    vals = []
    for sig in (1e6, 2e6):
        m = CosmologyAdvancedSampling(_lmax=lmax, _NSIDE=nside, _noisesig=sig,
                                      data_mode="synthetic", dtype=tf.complex128,
                                      use_matrixfree_sht=True, beam_fwhm_arcmin=600.0)
        m._ensure_tf_tensors()
        # Same (zero) data in both, so the likelihood is exactly sum(T^2)/(2 sig^2).
        m.prior_map_masked = tf.zeros(len(m.unmasked_idx), tf.float64)
        vals.append(float(psi_lensed(m, tf.constant(params, tf.float64), phi)))
    # likelihood ~ 1/sig^2: remove it with the two noise levels
    prior_plus_cl = (4 * vals[1] - vals[0]) / 3.0
    n_real = lmax * (lmax + 1) // 2 - 3
    a = np.asarray(splittosingularalm(params[lmax - 2:lmax - 2 + n_real],
                                      params[lmax - 2 + n_real:], lmax))
    w = np.array([1.0 if mm == 0 else 2.0 for L in range(lmax) for mm in range(L + 1)])
    ell = np.array([L for L in range(lmax) for mm in range(L + 1)])
    cl = np.concatenate([[1.0, 1.0], np.exp(params[:lmax - 2])])
    prior = 0.5 * np.sum(w * np.abs(a) ** 2 / (cl[ell] + 1e-30))
    entropy = np.sum((np.arange(lmax) + 0.5) * np.log(cl))
    assert prior_plus_cl == pytest.approx(prior + entropy, rel=1e-6)


# ---------------------------------------------------------------------------
# figure 2 / figure 3 on the smoke ensemble
# ---------------------------------------------------------------------------

def _pdf_width(path):
    import re
    m = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)", open(path, "rb").read())
    return float(m.group(3)) / 72.0


def _paper_path():
    p = os.path.join(SCRIPTS, "paper")
    if p not in sys.path:
        sys.path.insert(0, p)


def test_fig2_end_to_end_on_same_sky_pairs(smoke_ensemble, tmp_path, monkeypatch, capsys):
    _paper_path()
    import fig2_bias_reduction as f2
    import paper_style

    monkeypatch.setattr(sys, "argv", ["f2", "--indir", str(smoke_ensemble),
                                      "--outdir", str(tmp_path)])
    f2.main()
    out = capsys.readouterr().out
    assert "2 sky pairs" in out
    for name in ("bias_by_bin", "bias_per_sky"):
        assert _pdf_width(tmp_path / f"{name}.pdf") == pytest.approx(paper_style.FIG_1COL[0], abs=1e-3)


def test_fig2_expected_lensing_is_zero_without_phi(smoke_ensemble):
    """The 'lensing received' reference must vanish when phi does."""
    _paper_path()
    import fig2_bias_reduction as f2

    d = dict(np.load(smoke_ensemble / "chain_r001.npz"))
    b = dict(np.load(smoke_ensemble / "blind_r001.npz"))
    d["phi_true_packed"] = np.zeros_like(d["phi_true_packed"])
    b["phi_true_packed"] = d["phi_true_packed"]
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        np.savez(os.path.join(t, "chain_r000.npz"), **d)
        np.savez(os.path.join(t, "blind_r000.npz"), **b)
        rows, _ = f2.sky_biases(os.path.join(t, "chain_r000.npz"),
                                os.path.join(t, "blind_r000.npz"))
    exp = rows[:, 2]
    assert np.all(np.abs(exp[np.isfinite(exp)]) < 1e-6)


def test_fig2_refuses_a_pair_from_different_skies(smoke_ensemble, tmp_path):
    _paper_path()
    import fig2_bias_reduction as f2

    b = dict(np.load(smoke_ensemble / "blind_r001.npz"))
    b["alm_true_packed"] = b["alm_true_packed"] * 1.01
    np.savez(tmp_path / "chain_r000.npz", **dict(np.load(smoke_ensemble / "chain_r001.npz")))
    np.savez(tmp_path / "blind_r000.npz", **b)
    with pytest.raises(SystemExit, match="not the same sky"):
        f2.sky_biases(str(tmp_path / "chain_r000.npz"), str(tmp_path / "blind_r000.npz"))


def _fake_validation(path, passed=True, amp=AMP):
    L = np.arange(LMAX + 1)
    np.savez(path, passed=passed, lmax=LMAX, nside=NSIDE, noisesig=1.0,
             phi_amplitude=amp, fiducial="corrected", lensing_operator="exact",
             nl_qe=np.where(L >= 2, 1e-3, np.inf), median_ratio=1.0,
             median_response=1.0, n_sims=4)


def test_fig3_builds_with_a_matching_validation(smoke_ensemble, tmp_path, monkeypatch):
    _paper_path()
    import fig3_uncertainty_propagation as f3

    _fake_validation(tmp_path / "v.npz")
    monkeypatch.setattr(sys, "argv", ["f3", "--indir", str(smoke_ensemble), "--thin", "1",
                                      "--validation", str(tmp_path / "v.npz"),
                                      "--outdir", str(tmp_path / "out")])
    f3.main()
    for name in ("phi_uncertainty_vs_L", "phi_uncertainty_ratio"):
        assert (tmp_path / "out" / f"{name}.pdf").exists()


@pytest.mark.parametrize("kw", [{"passed": False}, {"amp": 1.0}])
def test_fig3_refuses_a_failed_or_mismatched_validation(smoke_ensemble, tmp_path, monkeypatch, kw):
    _paper_path()
    import fig3_uncertainty_propagation as f3

    _fake_validation(tmp_path / "v.npz", **kw)
    monkeypatch.setattr(sys, "argv", ["f3", "--indir", str(smoke_ensemble), "--thin", "1",
                                      "--validation", str(tmp_path / "v.npz"),
                                      "--outdir", str(tmp_path / "out")])
    with pytest.raises(SystemExit):
        f3.main()


def test_fig_maps_end_to_end_on_a_smoke_chain(smoke_ensemble, tmp_path, monkeypatch, capsys):
    """Replays the chain's data through its own operator and draws all six panels."""
    _paper_path()
    import fig_maps

    monkeypatch.setattr(sys, "argv", ["fm", "--chain", str(smoke_ensemble / "chain_r001.npz"),
                                      "--outdir", str(tmp_path), "--thin", "1"])
    fig_maps.main()
    out = capsys.readouterr().out
    assert "operator=exact" in out
    for name in ("map_true_unlensed", "map_observed_data", "map_lensing_signal",
                 "map_true_phi", "map_mean_phi", "map_residual_phi"):
        assert (tmp_path / f"{name}.pdf").exists()


# ---------------------------------------------------------------------------
# validate_coverage_rank_nulls: Block 4 exactness PIT
# ---------------------------------------------------------------------------

def _block4_chain(path, lmax=12, n=200, nu=6.0, seed=0):
    """phi draws that genuinely move, each with an EXACT Block 4 draw of C_L|phi."""
    from diffcmb.lensing import sample_cl_phiphi_given_phi

    rng = np.random.default_rng(seed)
    fid = np.zeros(lmax)
    fid[2:] = 1e-6 / np.arange(2, lmax) ** 2
    phi = rng.normal(size=(n, packed_length(lmax))) * 1e-3 * rng.uniform(0.3, 3.0, size=(n, 1))
    lnC = np.array([sample_cl_phiphi_given_phi(p, lmax, rng=rng, prior_nu=nu,
                                               cl_phiphi_fid=fid) for p in phi])
    np.savez(path, lmax=lmax, phi_samples=phi, cl_phiphi_samples=lnC,
             cl_phiphi_prior_nu=nu, cl_phiphi_fid=fid)


def test_block4_pit_is_uniform_for_exact_draws_and_breaks_when_misaligned(tmp_path):
    import validate_coverage_rank_nulls as v
    from scipy import stats

    _block4_chain(tmp_path / "chain_r000.npz")
    files = [str(tmp_path / "chain_r000.npz")]
    u0 = v._block4_u(files, stride=1, lag=0)
    assert stats.kstest(u0, "uniform").pvalue > 0.01
    u1 = v._block4_u(files, stride=1, lag=1)
    assert stats.kstest(u1, "uniform").pvalue < 1e-6


def test_block4_prior_is_read_from_the_chain(tmp_path):
    import validate_coverage_rank_nulls as v

    _block4_chain(tmp_path / "c.npz", nu=6.0)
    d = np.load(tmp_path / "c.npz")
    a0, b0 = v.prior_of(d, "cl_phiphi")
    assert a0 == 3.0 and np.allclose(b0, 6.0 * d["cl_phiphi_fid"])
    np.testing.assert_allclose(v.alpha_of(d, "cl_phiphi")[2:],
                               (2 * np.arange(2, 12) + 1 + 6.0) / 2.0)
    a0, b0 = v.prior_of(d, "cl_tt")
    assert a0 == -1.0 and not b0.any()


# ---------------------------------------------------------------------------
# validate_qe_noise: runs at small size and records its configuration
# ---------------------------------------------------------------------------

def test_validate_qe_noise_runs_and_records_configuration(tmp_path):
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "diffcmb"), OMP_NUM_THREADS="2")
    out = tmp_path / "qe.npz"
    res = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS, "validate_qe_noise.py"), "--lmax", "16",
         "--nside", "16", "--noisesig", "30", "--n_sims", "4", "--phi_amplitude", "300",
         "--out", str(out)], env=env, capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    d = np.load(out)
    assert float(d["phi_amplitude"]) == 300.0
    assert str(d["fiducial"]) == "corrected" and str(d["lensing_operator"]) == "exact"
    assert d["cl_tt_lensed"].shape == (17,) and np.all(d["cl_tt_lensed"][2:] > 0)
    assert np.all(np.isfinite(d["nl_qe"][2:16]))
    assert d["passed"].dtype == bool
