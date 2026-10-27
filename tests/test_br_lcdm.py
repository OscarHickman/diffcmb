"""scripts/br_lcdm.py: the Blackwell-Rao parameter likelihood from the chains' a_lm samples."""
import br_lcdm as br
import numpy as np
import pytest

from diffcmb.alm_utils import packed_length
from diffcmb.model import CosmologyAdvancedSampling

LMAX = 12


def _draw_packed(rng, cl, n):
    """n packed a_lm vectors drawn from C_l (Var Re = Var Im = C/2 for m >= 1, C for m = 0)."""
    w = br.sl_weights(LMAX)                                   # (P, LMAX), 1 or 2 per slot
    ell_of = np.argmax(w > 0, axis=1)
    var = cl[ell_of] / w[np.arange(w.shape[0]), ell_of]
    return rng.standard_normal((n, w.shape[0])) * np.sqrt(var)


def test_sl_matches_the_samplers_own_power():
    rng = np.random.default_rng(0)
    alm = rng.standard_normal((3, packed_length(LMAX)))
    dummy = type("M", (), {"lmax": LMAX})()
    want = np.stack([CosmologyAdvancedSampling.compute_sl_np(dummy, a) for a in alm])
    np.testing.assert_allclose(br.sl_from_packed(alm, LMAX), want, rtol=1e-12)


def test_s_over_c_has_2l_plus_1_degrees_of_freedom():
    rng = np.random.default_rng(1)
    cl = np.r_[0, 0, 1.0 / np.arange(2, LMAX) ** 2]
    s = br.sl_from_packed(_draw_packed(rng, cl, 20000), LMAX)
    k = br.dof(LMAX)
    np.testing.assert_allclose((s[:, 2:] / cl[2:]).mean(0), k[2:], rtol=0.03)


def test_one_exact_sample_under_the_flat_prior_peaks_at_the_amplitude_estimate():
    # exact a_lm (a delta posterior) and the flat chain prior: L(A) ~ A^(-sum k/2) e^(-sum S/C/2A),
    # whose maximum is A = sum(S/C) / sum(k).
    rng = np.random.default_rng(2)
    cl = np.r_[0, 0, 1.0 / np.arange(2, LMAX) ** 2]
    s = br.sl_from_packed(_draw_packed(rng, cl, 1), LMAX)
    amps = np.linspace(0.5, 1.5, 2001)
    grid = amps[:, None] * cl[None, 2:]
    ll, ess = br.log_like_grid(s[:, 2:], br.dof(LMAX)[2:], grid, br.chain_log_marginal(s[:, 2:], br.dof(LMAX)[2:]))
    want = (s[0, 2:] / cl[2:]).sum() / br.dof(LMAX)[2:].sum()
    assert amps[np.argmax(ll)] == pytest.approx(want, abs=1e-3)
    assert np.all(ess == 1.0)


def test_amplitude_posterior_is_calibrated_over_skies():
    # 200 skies at A = 1, each "chain" = the exact sky: the posterior CDF at the truth is uniform.
    rng = np.random.default_rng(3)
    cl = np.r_[0, 0, 1.0 / np.arange(2, LMAX) ** 2]
    amps = np.linspace(0.4, 2.0, 1601)
    grid = amps[:, None] * cl[None, 2:]
    k = br.dof(LMAX)[2:]
    u = []
    for _ in range(200):
        s = br.sl_from_packed(_draw_packed(rng, cl, 1), LMAX)[:, 2:]
        ll, _ = br.log_like_grid(s, k, grid, br.chain_log_marginal(s, k))
        post = np.exp(ll - ll.max())
        u.append(post[amps < 1.0].sum() / post.sum())
    u = np.array(u)
    assert abs(u.mean() - 0.5) < 0.06 and abs(u.std() - np.sqrt(1 / 12)) < 0.05


def _toy_gibbs(rng, d, slot_ell, w, noise, n_steps, nu=None, cl_fid=None):
    """Exact Gibbs on a noisy full sky, d = a + n per packed slot: a | C, d is Gaussian and
    C_l | a inverse-gamma under the flat prior (uniform in C) or InvGamma(nu/2, nu C_fid/2)."""
    k = br.dof(LMAX)[2:]
    cl = np.ones(LMAX)
    out = []
    for _ in range(n_steps):
        v, nv = cl[slot_ell] / w, noise[slot_ell] / w
        a = rng.normal(v / (v + nv) * d, np.sqrt(v * nv / (v + nv)))
        s = br.sl_from_packed(a[None], LMAX)[0, 2:]
        alpha = k / 2 - 1 if nu is None else (k + nu) / 2
        beta = s / 2 if nu is None else (s + nu * cl_fid) / 2
        cl[2:] = beta / rng.gamma(alpha)
        out.append(a)
    return np.array(out[n_steps // 5:])


def test_chains_under_either_c_prior_recover_the_exact_data_likelihood():
    # The reweighting divides out the chain's own C_l prior: a flat-prior chain and a nu = 30
    # chain on the same noisy sky give the same L(A), and both match the analytic p(d | A).
    rng = np.random.default_rng(5)
    cl = np.r_[0, 0, 1.0 / np.arange(2, LMAX) ** 2]
    noise = np.r_[0, 0, 0.3 * cl[2:]]
    w = br.sl_weights(LMAX)
    slot_ell = np.argmax(w > 0, axis=1)
    wslot = w[np.arange(w.shape[0]), slot_ell]
    d = (rng.standard_normal(w.shape[0]) * np.sqrt(1.2 * cl[slot_ell] / wslot)
         + rng.standard_normal(w.shape[0]) * np.sqrt(noise[slot_ell] / wslot))
    amps = np.linspace(0.6, 2.2, 321)
    grid = amps[:, None] * cl[None, 2:]
    va = amps[:, None] * cl[slot_ell][None] / wslot
    nv = noise[slot_ell] / wslot
    exact = -0.5 * (np.log(va + nv) + d ** 2 / (va + nv)).sum(1)
    k = br.dof(LMAX)[2:]

    def moments(ll):
        p = np.exp(ll - ll.max())
        p /= p.sum()
        m = (p * amps).sum()
        return m, np.sqrt((p * (amps - m) ** 2).sum())

    m0, s0 = moments(exact)
    for nu in (None, 30.0):
        fid = None if nu is None else cl[2:]
        a = _toy_gibbs(rng, d, slot_ell, wslot, noise, 6000, nu=nu, cl_fid=fid)
        s = br.sl_from_packed(a, LMAX)[:, 2:]
        ll, ess = br.log_like_grid(s, k, grid, br.chain_log_marginal(s, k, nu=nu, cl_fid=fid))
        m, sd = moments(ll)
        assert m == pytest.approx(m0, abs=0.15 * s0), nu
        assert sd == pytest.approx(s0, rel=0.15), nu


def test_hierarchical_target_is_calibrated_when_truths_scatter_around_the_model():
    # The strict ensembles draw each sky's C_l from InvGamma(nu/2, nu C_l(theta0)/2): under
    # --target hier the amplitude posterior must cover A = 1 uniformly; --target lcdm, which
    # ignores the scatter, is overconfident on the same skies (mildly at l < 12).
    rng = np.random.default_rng(6)
    nu = 30.0
    cl = np.r_[0, 0, 1.0 / np.arange(2, LMAX) ** 2]
    amps = np.linspace(0.3, 3.0, 2701)
    grid = amps[:, None] * cl[None, 2:]
    k = br.dof(LMAX)[2:]
    u = {"hier": [], "lcdm": []}
    for _ in range(300):
        c_true = np.r_[0, 0, nu * cl[2:] / 2 / rng.gamma(nu / 2, size=LMAX - 2)]
        s = br.sl_from_packed(_draw_packed(rng, c_true, 1), LMAX)[:, 2:]
        marg = br.chain_log_marginal(s, k)
        for target in u:
            ll, _ = br.log_like_grid(s, k, grid, marg, target, nu if target == "hier" else None)
            post = np.exp(ll - ll.max())
            u[target].append(post[amps < 1.0].sum() / post.sum())
    uh, ul = np.array(u["hier"]), np.array(u["lcdm"])
    assert abs(uh.mean() - 0.5) < 0.05 and abs(uh.std() - np.sqrt(1 / 12)) < 0.04
    assert ul.std() > uh.std()                        # spread towards 0 and 1


def test_per_multipole_ess_stays_usable_where_one_sample_would_not():
    # 500 samples of a noisy sky: every multipole's weights are healthy at the posterior peak.
    rng = np.random.default_rng(7)
    cl = np.r_[0, 0, 1.0 / np.arange(2, LMAX) ** 2]
    noise = np.r_[0, 0, 0.1 * cl[2:]]
    w = br.sl_weights(LMAX)
    slot_ell = np.argmax(w > 0, axis=1)
    wslot = w[np.arange(w.shape[0]), slot_ell]
    d = rng.standard_normal(w.shape[0]) * np.sqrt((cl + noise)[slot_ell] / wslot)
    a = _toy_gibbs(rng, d, slot_ell, wslot, noise, 700)
    s = br.sl_from_packed(a, LMAX)[:, 2:]
    k = br.dof(LMAX)[2:]
    ll, ess = br.log_like_grid(s, k, cl[None, 2:] * np.array([[1.0]]), br.chain_log_marginal(s, k))
    assert ess[0] > 0.3 * s.shape[0]


def test_model_grid_is_normalised_at_the_pivot_for_every_tilt():
    # A_pivot = C_pivot / C_pivot^fid: on a power-law tilt the model at the pivot is A x fid
    # whatever n_s (the pivot lies inside [LMIN, LMAX_USE), so read it off the grid).
    n = max(br.LMAX_USE, br.L_PIVOT + 1)
    ell = np.arange(n, dtype=float)
    ell[:2] = 1.0
    ratio = (ell[None] / 700.0) ** (br.NS_GRID[:, None] - br.NS_FID)
    base = np.ones(br.LMAX_USE - br.LMIN)
    g = br.model_grid(base, ratio).reshape(br.A_GRID.size, br.NS_GRID.size, -1)
    at_pivot = g[..., br.L_PIVOT - br.LMIN]
    np.testing.assert_allclose(at_pivot, np.broadcast_to(br.A_GRID[:, None], at_pivot.shape), rtol=1e-12)


def test_edge_fraction_flags_a_truncated_posterior():
    a, n = np.meshgrid(np.linspace(-4, 4, 81), np.linspace(-4, 4, 81), indexing="ij")
    inside = np.exp(-0.5 * (a ** 2 + n ** 2))
    assert br.edge_fraction(inside) < 1e-3
    assert br.edge_fraction(np.exp(-0.5 * ((a - 3.5) ** 2 + n ** 2))) > 0.5


def test_cdf_at_truth_is_one_half_for_a_posterior_centred_on_the_truth():
    a, n = np.meshgrid(br.A_GRID, br.NS_GRID, indexing="ij")
    p = np.exp(-0.5 * (((a - 1.0) / 0.03) ** 2 + ((n - br.NS_FID) / 0.05) ** 2))
    ua, un = br.cdf_at_truth(p / p.sum())
    assert ua == pytest.approx(0.5, abs=1e-3) and un == pytest.approx(0.5, abs=1e-3)
    p2 = np.exp(-0.5 * (((a - 1.06) / 0.03) ** 2 + ((n - br.NS_FID) / 0.05) ** 2))
    assert br.cdf_at_truth(p2 / p2.sum())[0] == pytest.approx(0.0228, abs=2e-3)   # truth 2 sd below
