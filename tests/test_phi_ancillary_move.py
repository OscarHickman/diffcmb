"""Tests for the ancillary (non-centred) joint (phi, C_L^phiphi) rescaling move.

Why this move exists
--------------------
Block 3 samples phi at fixed C_L^phiphi; Block 4 draws C_L^phiphi | phi. That
pair is a *centred* parameterisation of a variance-and-amplitude model, and it
funnels: job 11836793 (2026-08-24) measured worst lag-1 autocorrelation 0.945
with Block 4 on against 0.557 with it off, at an otherwise identical lmax=64
configuration that had passed the equilibration gate.

The move implemented here is the ancillary half of an ancillary-sufficient
interweaving strategy (Yu & Meng; the "ancillary vs sufficient
reparameterisation" of Millea, Anderes & Wandelt 2020, arXiv:2002.00965, and
the direct analogue of Racine et al. 2016's joint move for the (a_lm, C_l)
Gibbs funnel, arXiv:1512.06619). It holds xi = phi / sqrt(C) fixed and slides
(phi, C) along the funnel axis:

    phi -> alpha_L * phi ,   C_L -> alpha_L^2 * C_L

Because S_L(alpha*phi) = alpha^2 * S_L(phi), the Gaussian prior exponent
S_L / (2 C_L) is exactly invariant under this map -- that invariance is the
defining property and is asserted directly below.

Acceptance ratio
----------------
The composite chain must target the SAME joint density that Block 4's exact
conditional already implies. Block 4 draws C_L | phi ~ InvGamma(L - 0.5,
S_L / 2), which corresponds to

    log p(phi, C) = -psi(phi) - 0.5 * sum_L [ S_L(phi)/C_L + (2L+1) ln C_L ]

up to a constant, i.e. a flat (improper) prior on C_L. The packed phi vector
carries n_L = (L+1) real + (L-1) imaginary = 2L coordinates at multipole L, so
the map's Jacobian is prod_L alpha_L^(2L) * alpha_L^2. Combining,

    log A = -[psi(phi') - psi(phi)] + sum_L ln alpha_L

which the tests below verify against an independent brute-force evaluation
rather than trusting the derivation.
"""
import numpy as np
import pytest


def _has_deps():
    try:
        import healpy  # noqa: F401
        import scipy  # noqa: F401
        return True
    except Exception:
        return False


skip_no_deps = pytest.mark.skipif(not _has_deps(), reason="healpy/scipy unavailable")


from diffcmb.alm_utils import packed_sizes as _packed_sizes


def _coord_L_and_weight(lmax):
    """Multipole and S_L weight of every coordinate in the packed phi layout.

    Mirrors lensing.py::compute_sl_phi_np's traversal exactly: real parts for
    L=2..lmax-1 and m=0..L (weight 1 at m=0, else 2), then imaginary parts for
    m>=1 (weight 2). Written out independently here so the tests do not
    inherit an indexing bug from the module under test.
    """
    L_arr, w_arr = [], []
    for L in range(2, lmax):
        for m in range(L + 1):
            L_arr.append(L)
            w_arr.append(1.0 if m == 0 else 2.0)
    for L in range(2, lmax):
        for _m in range(1, L + 1):
            L_arr.append(L)
            w_arr.append(2.0)
    return np.array(L_arr), np.array(w_arr)


def _brute_force_log_target(phi, cl_full, lmax, neg_log_lik_fn):
    """log p(phi, C) up to an additive constant, from the definition in the
    module docstring. Deliberately a separate, naive implementation."""
    L_arr, w_arr = _coord_L_and_weight(lmax)
    total = -neg_log_lik_fn(phi)
    for L in range(2, lmax):
        sel = L_arr == L
        S_L = float(np.sum(w_arr[sel] * phi[sel] ** 2))
        C_L = float(cl_full[L])
        total += -0.5 * (S_L / C_L + (2 * L + 1) * np.log(C_L))
    return total


@skip_no_deps
def test_rescale_move_leaves_prior_exponent_exactly_invariant():
    """The defining property: S_L/C_L is unchanged by the move, for every L.

    If this fails the move is not sliding along the funnel axis and the whole
    construction is pointless, whatever the acceptance ratio says.
    """
    from diffcmb.lensing import compute_sl_phi_np, sample_phi_amplitude_rescale

    lmax = 6
    n_real, n_imag = _packed_sizes(lmax)
    rng = np.random.default_rng(0)
    phi = rng.normal(0.0, 1e-3, size=n_real + n_imag)
    cl_full = np.zeros(lmax)
    cl_full[2:] = 1e-6

    res = sample_phi_amplitude_rescale(
        phi, cl_full, lmax,
        neg_log_lik_fn=lambda p: 0.0,
        rng=np.random.default_rng(1),
        proposal_scale=0.4,
    )

    S_old = compute_sl_phi_np(phi, lmax)
    S_new = compute_sl_phi_np(res.phi_proposed, lmax)
    for L in range(2, lmax):
        np.testing.assert_allclose(
            S_new[L] / res.cl_phiphi_proposed[L],
            S_old[L] / cl_full[L],
            rtol=1e-12,
            err_msg=f"prior exponent S_L/C_L not invariant at L={L}",
        )


@skip_no_deps
def test_rescale_move_acceptance_ratio_matches_brute_force_target():
    """log_accept_ratio must equal the brute-force joint-target difference plus
    the analytic Jacobian -- the independent-reference discipline."""
    from diffcmb.lensing import sample_phi_amplitude_rescale

    lmax = 6
    n_real, n_imag = _packed_sizes(lmax)
    rng = np.random.default_rng(2)
    phi = rng.normal(0.0, 1e-3, size=n_real + n_imag)
    cl_full = np.zeros(lmax)
    cl_full[2:] = np.geomspace(1e-6, 1e-8, lmax - 2)
    data = rng.normal(0.0, 1e-3, size=n_real + n_imag)

    def neg_log_lik_fn(p):
        return 0.5 * float(np.sum((p - data) ** 2)) / (1e-3 ** 2)

    res = sample_phi_amplitude_rescale(
        phi, cl_full, lmax,
        neg_log_lik_fn=neg_log_lik_fn,
        rng=np.random.default_rng(3),
        proposal_scale=0.3,
    )

    lp_old = _brute_force_log_target(phi, cl_full, lmax, neg_log_lik_fn)
    lp_new = _brute_force_log_target(
        res.phi_proposed, res.cl_phiphi_proposed, lmax, neg_log_lik_fn
    )
    # Jacobian: n_L = 2L+1 phi coordinates plus the single C_L, which scales
    # as alpha^2 -> exponent (2L+1) + 2 per multipole.
    log_jac = sum((2 * L + 3) * res.log_alpha[L - 2] for L in range(2, lmax))
    expected = lp_new - lp_old + log_jac

    np.testing.assert_allclose(res.log_accept_ratio, expected, rtol=1e-9, atol=1e-9)


@skip_no_deps
def test_rescale_move_with_zero_proposal_scale_is_identity_and_always_accepts():
    from diffcmb.lensing import sample_phi_amplitude_rescale

    lmax = 5
    n_real, n_imag = _packed_sizes(lmax)
    rng = np.random.default_rng(4)
    phi = rng.normal(0.0, 1e-3, size=n_real + n_imag)
    cl_full = np.zeros(lmax)
    cl_full[2:] = 1e-6

    calls = []

    def neg_log_lik_fn(p):
        calls.append(p)
        return 0.5 * float(np.sum(p ** 2)) / (1e-3 ** 2)

    res = sample_phi_amplitude_rescale(
        phi, cl_full, lmax,
        neg_log_lik_fn=neg_log_lik_fn,
        rng=np.random.default_rng(5),
        proposal_scale=0.0,
    )

    assert res.accepted
    np.testing.assert_allclose(res.log_alpha, 0.0, atol=0.0)
    np.testing.assert_allclose(res.phi, phi, rtol=0, atol=0)
    np.testing.assert_allclose(res.cl_phiphi, cl_full, rtol=0, atol=0)
    np.testing.assert_allclose(res.log_accept_ratio, 0.0, atol=1e-12)


@skip_no_deps
def test_rescale_move_rejects_and_returns_current_state_unchanged():
    """A guaranteed-reject move must hand back the *current* state, not the
    proposal -- the failure mode that would silently corrupt a chain."""
    from diffcmb.lensing import sample_phi_amplitude_rescale

    lmax = 5
    n_real, n_imag = _packed_sizes(lmax)
    rng = np.random.default_rng(6)
    phi = rng.normal(0.0, 1e-3, size=n_real + n_imag)
    cl_full = np.zeros(lmax)
    cl_full[2:] = 1e-6

    # An astronomically steep likelihood at the current point makes any
    # nonzero rescaling infinitely worse, so the move must always reject.
    def neg_log_lik_fn(p):
        return 0.0 if np.array_equal(p, phi) else 1e300

    res = sample_phi_amplitude_rescale(
        phi, cl_full, lmax,
        neg_log_lik_fn=neg_log_lik_fn,
        rng=np.random.default_rng(7),
        proposal_scale=0.5,
    )

    assert not res.accepted
    np.testing.assert_allclose(res.phi, phi, rtol=0, atol=0)
    np.testing.assert_allclose(res.cl_phiphi, cl_full, rtol=0, atol=0)


@skip_no_deps
def test_rescale_move_preserves_stationary_distribution_of_exact_gibbs_chain():
    """The invariance test that actually matters.

    On a tractable model whose two Gibbs conditionals are known exactly, adding
    the rescaling move to an already-exact chain must not move the stationary
    distribution. Reference chain = exact Gibbs alone; test chain = exact Gibbs
    + the move. If the acceptance ratio or Jacobian is wrong, the test chain's
    C_L marginal shifts and the comparison fails.

    Model: phi ~ N(0, C) in the packed S_L metric (the code's own prior),
    d | phi ~ N(phi, sigma^2 I), flat prior on C. Then
      C_L | phi ~ InvGamma(L - 0.5, S_L/2)                (Block 4, exact)
      phi_i | C, d ~ N(mu_i, 1/prec_i),
          prec_i = 1/sigma^2 + w_i / C_{L_i},  mu_i = (d_i/sigma^2)/prec_i
    """
    from scipy import stats

    from diffcmb.lensing import (
        sample_cl_phiphi_given_phi,
        sample_phi_amplitude_rescale,
    )

    lmax = 5
    n_real, n_imag = _packed_sizes(lmax)
    n = n_real + n_imag
    L_arr, w_arr = _coord_L_and_weight(lmax)
    sigma = 1.0e-3
    data = np.random.default_rng(10).normal(0.0, 1.0e-3, size=n)

    def neg_log_lik_fn(p):
        return 0.5 * float(np.sum((p - data) ** 2)) / sigma ** 2

    def draw_phi_given_cl(cl_full, rng):
        prec = 1.0 / sigma ** 2 + w_arr / cl_full[L_arr]
        mu = (data / sigma ** 2) / prec
        return rng.normal(mu, 1.0 / np.sqrt(prec))

    def run_chain(with_move, seed, n_iter=6000):
        rng = np.random.default_rng(seed)
        cl_full = np.zeros(lmax)
        cl_full[2:] = 1.0e-6
        phi = draw_phi_given_cl(cl_full, rng)
        trace = []
        for it in range(n_iter):
            cl_full = cl_full.copy()
            cl_full[2:lmax] = np.exp(sample_cl_phiphi_given_phi(phi, lmax, rng))
            phi = draw_phi_given_cl(cl_full, rng)
            if with_move:
                res = sample_phi_amplitude_rescale(
                    phi, cl_full, lmax,
                    neg_log_lik_fn=neg_log_lik_fn,
                    rng=rng,
                    proposal_scale=0.25,
                )
                phi, cl_full = res.phi, res.cl_phiphi
            if it >= 1000:
                trace.append(np.log(cl_full[2]))
        return np.array(trace)

    ref = run_chain(with_move=False, seed=100)
    tst = run_chain(with_move=True, seed=200)

    # Thin hard before the KS test: consecutive Gibbs draws are autocorrelated,
    # so an unthinned KS p-value is anti-conservative and would make this test
    # flaky rather than sensitive.
    ref_t, tst_t = ref[::10], tst[::10]
    p = float(stats.ks_2samp(ref_t, tst_t).pvalue)

    assert p > 0.01, (
        f"stationary distribution of ln C_2 shifted when the rescaling move was "
        f"added (KS p={p:.4g}); the move does not leave the target invariant. "
        f"ref mean={ref.mean():.4f} sd={ref.std():.4f}; "
        f"test mean={tst.mean():.4f} sd={tst.std():.4f}"
    )
    # A KS test can pass on badly-estimated tails; pin the first two moments too.
    np.testing.assert_allclose(tst.mean(), ref.mean(), atol=0.15)
    np.testing.assert_allclose(tst.std(), ref.std(), rtol=0.20)


@skip_no_deps
def test_rescale_move_actually_accepts_sometimes_on_a_realistic_target():
    """A move that never accepts is invariant but useless -- guard against
    shipping a preconditioner-shaped no-op."""
    from diffcmb.lensing import (
        sample_cl_phiphi_given_phi,
        sample_phi_amplitude_rescale,
    )

    lmax = 5
    n_real, n_imag = _packed_sizes(lmax)
    n = n_real + n_imag
    L_arr, w_arr = _coord_L_and_weight(lmax)
    sigma = 1.0e-3
    rng = np.random.default_rng(11)
    data = rng.normal(0.0, 1.0e-3, size=n)

    def neg_log_lik_fn(p):
        return 0.5 * float(np.sum((p - data) ** 2)) / sigma ** 2

    cl_full = np.zeros(lmax)
    cl_full[2:] = 1.0e-6
    prec = 1.0 / sigma ** 2 + w_arr / cl_full[L_arr]
    phi = rng.normal((data / sigma ** 2) / prec, 1.0 / np.sqrt(prec))

    n_accept = 0
    for _ in range(300):
        cl_full = cl_full.copy()
        cl_full[2:lmax] = np.exp(sample_cl_phiphi_given_phi(phi, lmax, rng))
        res = sample_phi_amplitude_rescale(
            phi, cl_full, lmax,
            neg_log_lik_fn=neg_log_lik_fn,
            rng=rng,
            proposal_scale=0.1,
        )
        phi, cl_full = res.phi, res.cl_phiphi
        n_accept += int(res.accepted)

    assert n_accept > 15, (
        f"only {n_accept}/300 rescaling moves accepted at proposal_scale=0.1 -- "
        "the move is effectively a no-op"
    )
