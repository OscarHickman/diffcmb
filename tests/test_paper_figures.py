"""Tests for the paper-figure workflow (scripts/paper/) and the operator check.

`scripts/` is production infrastructure, not unit-tested by the rest of the
suite -- which is how the 2026-09-21 figures shipped an alm-ordering bug, a
fabricated burn-in line, a wrong Block 4 formula and pooled rank tests that
assumed 96 independent ranks. Each test below pins one of those, or one piece
of the statistics the figures print, at small size (no production chains, no
CAMB, no TensorFlow model).

Layout of the checks:
  * paper_style      -- journal presets, exact saved page size
  * fig_maps         -- packed -> healpy ordering on the sky (the stripe bug)
  * fig_convergence  -- rank-normalised R-hat and bulk ESS against known cases
  * fig1_validation  -- per-bin rank tests: calibrated under a correct sampler,
                        powerful against a biased one; end-to-end on a
                        synthetic ensemble
  * fig_schematic    -- its printed conditionals match the code's shapes
  * lensing power transfer -- the operator check behind ROADMAP D3
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import pytest

hp = pytest.importorskip("healpy")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("scripts", os.path.join("scripts", "paper")):
    path = os.path.join(REPO, sub)
    if path not in sys.path:
        sys.path.insert(0, path)

import paper_style  # noqa: E402

from diffcmb.alm_utils import (  # noqa: E402
    invgamma_shape_for_spectrum,
    packed_length,
    packed_sizes,
)
from diffcmb.lensing import _alm_hp_to_packed  # noqa: E402


def _pdf_size_inches(path):
    data = open(path, "rb").read()
    m = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", data)
    return float(m.group(3)) / 72.0, float(m.group(4)) / 72.0


# ---------------------------------------------------------------------------
# paper_style
# ---------------------------------------------------------------------------

def test_journal_preset_widths_prd_default():
    if paper_style.JOURNAL != "PRD":
        pytest.skip("DIFFCMB_JOURNAL overridden in this environment")
    assert paper_style.FIG_1COL[0] == pytest.approx(3.375)
    assert paper_style.FIG_2COL[0] == pytest.approx(7.0)
    assert 3 * paper_style.FIG_MAP[0] < paper_style.FIG_2COL[0]


def test_saved_panel_is_exactly_its_figsize(tmp_path):
    """Constrained layout + standard bbox: the page size is the printed size.

    With savefig.bbox='tight' (the old setting) panels came out 3.02-3.15 in
    wide instead of the column width, so no panel was at its stated size.
    """
    import matplotlib.pyplot as plt

    paper_style.apply()
    fig, ax = plt.subplots(figsize=paper_style.FIG_1COL)
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("a fairly long axis label to tempt a tight bbox")
    out = tmp_path / "panel.pdf"
    paper_style.save(fig, str(out))
    w, h = _pdf_size_inches(out)
    assert w == pytest.approx(paper_style.FIG_1COL[0], abs=1e-3)
    assert h == pytest.approx(paper_style.FIG_1COL[1], abs=1e-3)


# ---------------------------------------------------------------------------
# fig_maps: ordering on the sky
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ell,m", [(5, 3), (7, 0), (9, 9)])
def test_to_map_puts_a_single_mode_at_the_right_multipole(ell, m):
    """A packed vector with power in one (l, m) must map to that (l, m).

    fig_maps originally sent author-ordered alms straight to hp.alm2map, which
    reads m-major healpy order: every coefficient landed at the wrong (l, m)
    and the phi maps came out as zonal stripes.
    """
    import fig_maps

    lmax, nside = 12, 16
    lh = lmax - 1
    alm = np.zeros(hp.Alm.getsize(lh), dtype=complex)
    alm[hp.Alm.getidx(lh, ell, m)] = 1.0 if m == 0 else 1.0 + 0.5j
    packed = _alm_hp_to_packed(alm, lmax)
    back = hp.map2alm(fig_maps.to_map(packed, lmax, nside), lmax=lh, iter=5)
    power = np.abs(back) ** 2
    peak = int(np.argmax(power))
    assert peak == hp.Alm.getidx(lh, ell, m)
    assert power.sum() - power[peak] < 1e-6 * power[peak]


def test_mode_z_is_unit_normal_for_a_calibrated_posterior():
    import fig_maps

    rng = np.random.default_rng(1)
    n_draw, n_coord = 400, 3000
    mu = rng.normal(size=n_coord)
    sd = rng.uniform(0.5, 2.0, size=n_coord)
    truth = mu + sd * rng.normal(size=n_coord)
    samples = mu + sd * rng.normal(size=(n_draw, n_coord))
    z = fig_maps.mode_z(samples, truth)
    assert abs(z.mean()) < 0.05
    assert z.std() == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# fig_convergence: R-hat and ESS
# ---------------------------------------------------------------------------

def _ar1(n, rho, k, rng):
    x = np.empty((n, k))
    x[0] = rng.normal(size=k)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + np.sqrt(1 - rho ** 2) * rng.normal(size=k)
    return x


def test_rank_rhat_near_one_for_iid_and_large_for_a_shift():
    import fig_convergence as fc

    rng = np.random.default_rng(2)
    iid = rng.normal(size=(1000, 200))
    assert np.median(fc.rank_rhat(iid)) < 1.005
    drift = iid.copy()
    drift[500:] += 1.0                      # second half sits elsewhere
    assert np.all(fc.rank_rhat(drift) > 1.1)


def test_rank_rhat_folded_part_catches_a_scale_change():
    import fig_convergence as fc

    rng = np.random.default_rng(3)
    x = rng.normal(size=(2000, 100))
    x[1000:] *= 3.0                         # same mean, different width
    assert np.median(fc.rank_rhat(x)) > 1.05


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_bulk_ess_matches_ar1_theory(rho):
    import fig_convergence as fc

    rng = np.random.default_rng(4)
    n = 4000
    ess = fc.bulk_ess(_ar1(n, rho, 64, rng))
    expected = n * (1 - rho) / (1 + rho)
    assert np.median(ess) == pytest.approx(expected, rel=0.25)


# ---------------------------------------------------------------------------
# fig1_validation: per-bin rank tests
# ---------------------------------------------------------------------------

def test_mean_sd_p_is_calibrated_under_uniform_ranks():
    """False-positive rate at alpha=0.05 must be ~5% for exact uniform ranks."""
    import fig1_validation as f1

    rng = np.random.default_rng(5)
    n_draws, trials = 59, 200
    rejections = 0
    for _ in range(trials):
        r = rng.integers(0, n_draws + 1, size=24)
        p_m, p_s = f1.mean_sd_p(r, n_draws, n_rep=2000, seed=int(rng.integers(1e9)))
        rejections += (p_m < 0.05)
    assert 0.01 <= rejections / trials <= 0.10


def test_mean_sd_p_detects_bias_and_overdispersion():
    import fig1_validation as f1

    n_draws = 59
    low = np.full(24, 5)                                 # truth always near the bottom
    assert f1.mean_sd_p(low, n_draws, n_rep=4000)[0] < 0.01
    u_shape = np.array([0, 1, 58, 59] * 6)               # too-narrow posterior
    assert f1.mean_sd_p(u_shape, n_draws, n_rep=4000)[1] < 0.01


def test_binned_test_applies_the_multiplicity_correction():
    import fig1_validation as f1

    rng = np.random.default_rng(6)
    sets = [rng.integers(0, 60, size=24) for _ in range(4)]
    p_corr, p_all = f1.binned_test(sets, 59)
    assert len(p_all) == 8
    assert p_corr == pytest.approx(min(1.0, 8 * min(p_all)))


LMAX_SYN = 12


def _write_synthetic_ensemble(dirpath, n_chains, n_samp, bias=0.0, seed=7, lmax=None):
    """Chains whose truth is exchangeable with the draws (a 'correct sampler').

    `bias` shifts every draw of every field upward by that many posterior sd,
    which is what a biased sampler looks like to the rank test.
    """
    rng = np.random.default_rng(seed)
    lmax = LMAX_SYN if lmax is None else lmax
    n_pk, n_cl = packed_length(lmax), lmax - 2
    for c in range(n_chains):
        def field():
            mu = rng.normal(size=n_pk)
            truth = mu + rng.normal(size=n_pk)
            draws = mu + bias + rng.normal(size=(n_samp, n_pk))
            return truth, draws

        phi_t, phi_s = field()
        alm_t, alm_s = field()
        lnc_mu = rng.normal(size=n_cl)
        cpp_t = np.zeros(lmax)
        cpp_t[2:] = np.exp(lnc_mu + 0.3 * rng.normal(size=n_cl))
        cpp_s = lnc_mu + 0.3 * bias + 0.3 * rng.normal(size=(n_samp, n_cl))
        np.savez(
            os.path.join(dirpath, f"chain_r{c:03d}.npz"),
            lmax=lmax, nside=8, realization=c, packing_version=2,
            alm_samples=np.hstack([rng.normal(size=(n_samp, n_cl)), alm_s]),
            phi_samples=phi_s, alm_true_packed=alm_t, phi_true_packed=phi_t,
            cl_phiphi_samples=cpp_s, cl_phiphi_true=cpp_t,
            cl_phiphi_prior_nu=6.0, sample_cl_phiphi=True,
            logp=rng.normal(size=n_samp), n_burnin=10,
        )


def _run_fig1(indir, outdir, capsys, monkeypatch):
    import fig1_validation as f1

    monkeypatch.setattr(sys, "argv", ["fig1", "--indir", str(indir), "--outdir",
                                      str(outdir), "--thin", "2", "--n_rep", "50"])
    f1.main()
    out = capsys.readouterr().out
    return {tag: float(re.search(rf"{tag}.*?corrected min ([\d.]+)", out).group(1))
            for tag in ("phi_power", "alm_power", "clpp")}


def test_fig1_end_to_end_passes_a_correct_sampler(tmp_path, capsys, monkeypatch):
    ens = tmp_path / "ens"
    ens.mkdir()
    _write_synthetic_ensemble(ens, n_chains=24, n_samp=120)
    p = _run_fig1(ens, tmp_path / "out", capsys, monkeypatch)
    assert all(v > 0.01 for v in p.values()), p
    for name in ("rank_histograms", "clpp_sbc", "validation_power"):
        assert (tmp_path / "out" / f"{name}.pdf").exists()


def test_fig1_end_to_end_rejects_a_biased_sampler(tmp_path, capsys, monkeypatch):
    ens = tmp_path / "ens"
    ens.mkdir()
    _write_synthetic_ensemble(ens, n_chains=24, n_samp=120, bias=1.0)
    p = _run_fig1(ens, tmp_path / "out", capsys, monkeypatch)
    assert p["clpp"] < 0.01, p


def test_fig_convergence_end_to_end_on_synthetic_ensemble(tmp_path, monkeypatch, capsys):
    import fig_convergence as fc

    ens = tmp_path / "ens"
    ens.mkdir()
    _write_synthetic_ensemble(ens, n_chains=4, n_samp=100)
    monkeypatch.setattr(sys, "argv", ["fc", "--indir", str(ens), "--outdir",
                                      str(tmp_path / "out")])
    fc.main()
    out = capsys.readouterr().out
    assert "after 10 burn-in" in out          # read from the chain, not assumed
    for name in ("convergence_trace_logp", "convergence_rhat", "convergence_autocorr_ess"):
        w, h = _pdf_size_inches(tmp_path / "out" / f"{name}.pdf")
        assert (w, h) == pytest.approx(paper_style.FIG_1COL, abs=1e-3)


def test_fig_convergence_refuses_a_block4_off_ensemble(tmp_path):
    import fig_convergence as fc

    ens = tmp_path / "ens"
    ens.mkdir()
    _write_synthetic_ensemble(ens, n_chains=2, n_samp=20)
    f = ens / "chain_r000.npz"
    d = dict(np.load(f))
    d["sample_cl_phiphi"] = False
    np.savez(f, **d)
    with pytest.raises(SystemExit):
        fc.load(str(ens))


# ---------------------------------------------------------------------------
# fig_schematic: the conditionals it prints are the ones the code draws
# ---------------------------------------------------------------------------

def test_schematic_conditionals_match_code_shapes():
    """Pin the two InvGamma shapes the schematic prints to the code.

    Block 1 (flat prior): (2l+1)/2 - 1.  Block 4 (proper prior nu): (2L+1+nu)/2.
    The first schematic printed an extra '-1' in Block 4.
    """
    lmax, nu = 16, 6.0
    ells = np.arange(2, lmax)
    np.testing.assert_allclose(invgamma_shape_for_spectrum(lmax)[ells],
                               (2 * ells + 1) / 2 - 1)
    np.testing.assert_allclose(invgamma_shape_for_spectrum(lmax, a0=nu / 2)[ells],
                               (2 * ells + 1 + nu) / 2)
    src = open(os.path.join(REPO, "scripts", "paper", "fig_schematic.py")).read()
    # Anchor on the separator after the shape so an appended "-1" cannot hide.
    assert r"\frac{2L+1+\nu}{2},\," in src
    assert r"\frac{2\ell+1}{2}-1,\," in src
    assert "MCLMC" not in src.split('"""', 2)[2]      # outside the docstring


def test_schematic_builds_at_figure_star_width(tmp_path):
    import fig_schematic

    fig_schematic.make(str(tmp_path))
    w, _ = _pdf_size_inches(tmp_path / "schematic_forward_model.pdf")
    assert w == pytest.approx(paper_style.FIG_2COL[0], abs=1e-3)


# ---------------------------------------------------------------------------
# Lensing operator power transfer (ROADMAP D3)
# ---------------------------------------------------------------------------

def _toy_spectra(lmax_out):
    ell = np.arange(lmax_out, dtype=float)
    cl_unl = np.zeros(lmax_out)
    cl_unl[2:] = 1e3 / (ell[2:] * (ell[2:] + 1))
    cl_pp = np.zeros(lmax_out)
    cl_pp[2:] = 1e-7 / (ell[2:] * (ell[2:] + 1)) ** 2
    return cl_unl, cl_pp


def test_power_transfer_is_identity_without_lensing():
    import validate_lensing_power_transfer as v

    lmax = 16
    cl_unl, cl_pp = _toy_spectra(3 * lmax)
    res = v.run_lmax(lmax, 1, cl_unl, np.zeros_like(cl_pp), 2, 1, 0)
    for key in ("A", "B", "Bh"):
        np.testing.assert_allclose(res[key][2:], 1.0, atol=1e-6, err_msg=key)


def test_bilinear_operator_loses_more_power_than_exact_lensing():
    """The D3 finding at toy size: interpolation error dominates the lensing.

    With deflections of order a tenth of a pixel, exact evaluation at the same
    angles barely moves in-band power, while the production bilinear operator
    removes power that grows toward lmax.
    """
    import validate_lensing_power_transfer as v

    lmax = 16
    cl_unl, cl_pp = _toy_spectra(3 * lmax)
    res = v.run_lmax(lmax, 4, cl_unl, cl_pp * 1e3, 2, 1, 0)
    hi = slice(lmax // 2, lmax)
    loss_A = 1.0 - res["A"][hi].mean()
    loss_B = abs(1.0 - res["B"][hi].mean())
    assert loss_A > 0.0
    assert loss_A > 3.0 * loss_B
    np.testing.assert_allclose(res["B"][2:], res["Bh"][2:], rtol=0.02)


def test_binned_power_transfer_bins_are_weighted_means():
    import validate_lensing_power_transfer as v

    ratio = np.ones(64)
    ratio[10:30] = 1.02
    rows = v.binned(ratio, 64, 2.0 * np.arange(64) + 1.0)
    assert rows[0] == pytest.approx(0.0)
    assert rows[1] == pytest.approx(2.0)
    assert np.isnan(rows[-1])                  # [160,192) is above lmax=64


def test_packed_sizes_consistent_with_packed_length():
    n_real, n_imag = packed_sizes(LMAX_SYN)
    assert n_real + n_imag == packed_length(LMAX_SYN)


def test_fig4_end_to_end_on_a_synthetic_lmax64_ensemble(tmp_path, monkeypatch, capsys):
    """Independent draws: the joint C_l-C_L^phiphi correlation must be a null."""
    import fig4_joint_posterior as f4

    ens = tmp_path / "ens"
    ens.mkdir()
    _write_synthetic_ensemble(ens, n_chains=6, n_samp=60, lmax=64)
    monkeypatch.setattr(sys, "argv", ["f4", "--indir", str(ens), "--thin", "1",
                                      "--n_rep", "200", "--outdir", str(tmp_path / "out")])
    f4.main()
    out = capsys.readouterr().out
    n_above = int(re.search(r"(\d+)/16 cells above", out).group(1))
    assert n_above <= 3
    for name in ("joint_posterior_scatter", "joint_posterior_heatmap"):
        w, h = _pdf_size_inches(tmp_path / "out" / f"{name}.pdf")
        assert (w, h) == pytest.approx(paper_style.FIG_1COL_TALL, abs=1e-3)


# ---------------------------------------------------------------------------
# fig1_validation: the a_lm row against the exact-sampler null (ROADMAP T0.1c)
# ---------------------------------------------------------------------------

def test_mean_sd_p_against_a_null_pool_is_calibrated_and_has_power():
    """With a supplied null pool, ranks drawn FROM that pool must pass at ~5%,
    and exactly-uniform ranks must be rejected when the pool sits at 0.3."""
    import fig1_validation as f1

    rng = np.random.default_rng(8)
    n_draws, trials = 119, 200
    pool_r = np.clip(rng.normal(0.3, 0.2, size=4000) * (n_draws + 1), 0, n_draws)
    pool_u = (np.floor(pool_r) + 0.5) / (n_draws + 1.0)
    rejections = 0
    for _ in range(trials):
        r = np.rint(rng.choice(pool_u, 24) * (n_draws + 1) - 0.5).astype(int)
        p_m, _ = f1.mean_sd_p(r, n_draws, n_rep=2000,
                              seed=int(rng.integers(1e9)), null_u=pool_u)
        rejections += (p_m < 0.05)
    assert 0.01 <= rejections / trials <= 0.10
    uniform = rng.integers(0, n_draws + 1, size=24)
    assert f1.mean_sd_p(uniform, n_draws, n_rep=4000, null_u=pool_u)[0] < 0.01


def test_load_alm_null_refuses_a_draw_count_mismatch(tmp_path):
    """A null computed at another thinning lives on a different rank grid;
    comparing against it would be silently wrong, so the loader must refuse."""
    import fig1_validation as f1

    u = (np.arange(60) + 0.5) / 60.0                       # 59 draws + 1
    np.savez(tmp_path / f1.ALM_NULL_FILE, effective_2_10_u=u, effective_2_10_obs=0.5)
    assert f1.load_alm_null(str(tmp_path), 59, "effective", lmax=10) is not None
    with pytest.raises(SystemExit, match="draw"):
        f1.load_alm_null(str(tmp_path), 119, "effective", lmax=10)
    assert f1.load_alm_null(str(tmp_path / "missing"), 59, "effective", lmax=10) is None


def _alm_ranks_of_synthetic_ensemble(dirpath, n_chains, n_samp, bias, seed, thin):
    import fig1_validation as f1

    dirpath.mkdir()
    _write_synthetic_ensemble(dirpath, n_chains=n_chains, n_samp=n_samp,
                              bias=bias, seed=seed)
    records, n_draws = f1.collect_field_ranks(f1.chain_files(str(dirpath)), thin)
    return {k[1:]: np.asarray(v) for k, v in records.items() if k[0] == "alm_power"}, n_draws


def test_fig1_alm_row_passes_an_offset_that_the_null_predicts(tmp_path, capsys, monkeypatch):
    """A sampler whose a_lm ranks are offset EXACTLY as its own exact null says
    must pass against that null and fail against uniform -- the T0.1c logic."""
    ens = tmp_path / "ens"
    _alm_ranks_of_synthetic_ensemble(ens, 24, 120, bias=1.0, seed=7, thin=2)
    null_r, n_draws = _alm_ranks_of_synthetic_ensemble(
        tmp_path / "null", 300, 120, bias=1.0, seed=99, thin=2)
    import fig1_validation as f1
    np.savez(ens / f1.ALM_NULL_FILE,
             **{f"effective_{lo}_{hi}_u": (r + 0.5) / (n_draws + 1.0)
                for (lo, hi), r in null_r.items()})

    p_null = _run_fig1(ens, tmp_path / "o1", capsys, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["fig1", "--indir", str(ens), "--outdir",
                                      str(tmp_path / "o2"), "--thin", "2",
                                      "--n_rep", "50", "--alm_null", "uniform"])
    f1.main()
    out = capsys.readouterr().out
    p_unif = float(re.search(r"alm_power.*?corrected min ([\d.]+)", out).group(1))
    assert p_null["alm_power"] > 0.01, p_null
    assert p_unif < 0.01, p_unif
