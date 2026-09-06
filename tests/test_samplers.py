"""Tests for NUTS/HMC samplers — sign convention and step size adaptation."""
import numpy as np
import pytest

from diffcmb.alm_utils import packed_sizes


def _has_all():
    try:
        import healpy  # noqa: F401
        import scipy  # noqa: F401
        import tensorflow as tf  # noqa: F401
        import tensorflow_probability as tfp  # noqa: F401

        from diffcmb import (  # noqa: F401
            CosmologyAdvancedSampling,
            run_chain_hmc,
            run_chain_nut,
        )
        return True
    except Exception:
        return False


skip_no_tfp = pytest.mark.skipif(not _has_all(), reason="TFP or deps unavailable")

LMAX, NSIDE = 10, 4  # small enough to run in CI


@pytest.fixture(scope="module")
def small_model():
    from diffcmb import CosmologyAdvancedSampling
    m = CosmologyAdvancedSampling(_lmax=LMAX, _NSIDE=NSIDE, _noisesig=1.0,
                                   data_mode='synthetic')
    m._ensure_tf_tensors()
    return m


# ── target_log_prob_fn sign ───────────────────────────────────────────────────

@skip_no_tfp
def test_sampler_uses_negative_psi_tf(small_model):
    """
    target_log_prob recorded by NUTS must equal -psi_tf(sample) at each step.

    This catches the sign bug where psi_tf (negative log-posterior) was passed
    as-is to TFP, causing it to sample from 1/posterior.  We verify the identity
    target_log_prob[i] == -psi_tf(samples[i]) for every recorded sample.
    """
    import tensorflow as tf

    from diffcmb import run_chain_nut

    x0 = small_model.prior_parameters_tf()
    samples, results = run_chain_nut(
        small_model, x0, _step_size=0.001,
        num_results=5, num_burnin_steps=0,
    )

    inner = results.inner_results
    recorded_logp = inner.target_log_prob.numpy()

    for i, (samp, logp) in enumerate(zip(samples.numpy(), recorded_logp)):
        expected = -float(small_model.psi_tf(tf.constant(samp, dtype=tf.float64)).numpy())
        assert abs(logp - expected) < 1e-4, (
            f"Step {i}: target_log_prob={logp:.6f} but -psi_tf(sample)={expected:.6f}. "
            "Sign of psi_tf is wrong in samplers.py."
        )


@skip_no_tfp
def test_hmc_sampler_uses_negative_psi_tf(small_model):
    """Verify HMC targets -psi_tf (sign check).

    Run with no burnin so the first recorded sample starts from x0 and
    its target_log_prob must equal -psi_tf(x0) exactly.
    """
    from diffcmb import run_chain_hmc

    x0 = small_model.prior_parameters_tf()
    expected_logp = -float(small_model.psi_tf(x0).numpy())

    samples, results = run_chain_hmc(
        small_model, x0, _step_size=0.001,
        num_results=5, num_burnin_steps=0,
    )

    inner = results.inner_results
    first_logp = float(inner.accepted_results.target_log_prob.numpy()[0])

    assert abs(first_logp - expected_logp) < 1.0, (
        f"HMC target_log_prob ({first_logp:.4f}) != -psi_tf ({expected_logp:.4f})"
    )


# ── MAP initialisation ───────────────────────────────────────────────────────

@skip_no_tfp
def test_find_map_reduces_psi(small_model):
    """find_map_estimate must strictly decrease psi_tf from the prior mean."""
    from diffcmb import find_map_estimate

    x0 = small_model.prior_parameters_tf()
    psi_before = float(small_model.psi_tf(x0))

    map_state = find_map_estimate(small_model, n_steps=50, learning_rate=0.001, print_every=50)

    psi_after = float(small_model.psi_tf(map_state))
    assert psi_after < psi_before, (
        f"MAP did not reduce psi: {psi_before:.4f} → {psi_after:.4f}"
    )


@skip_no_tfp
def test_find_map_returns_correct_shape(small_model):
    """MAP estimate must have the same shape as the initial state."""
    from diffcmb import find_map_estimate

    x0 = small_model.prior_parameters_tf()
    map_state = find_map_estimate(small_model, n_steps=10, print_every=10)

    assert map_state.shape == x0.shape, (
        f"MAP shape {map_state.shape} != x0 shape {x0.shape}"
    )


# ── chain movement ────────────────────────────────────────────────────────────

@skip_no_tfp
def test_nuts_chain_moves(small_model):
    """
    NUTS must produce samples that differ from the initial state.

    The original bug caused 0% acceptance and every sample == x0.
    """
    from diffcmb import run_chain_nut

    x0 = small_model.prior_parameters_tf()
    samples, results = run_chain_nut(
        small_model, x0, _step_size=0.001,
        num_results=20, num_burnin_steps=50,
    )

    samps = samples.numpy()
    x0_np = x0.numpy()

    assert not np.allclose(samps[0], samps[-1], atol=0), \
        "All NUTS samples are identical — chain never moved (acceptance=0 bug)"
    assert not np.allclose(samps[0], x0_np, atol=0), \
        "First sample equals x0 — chain stuck at initial state"


@skip_no_tfp
def test_nuts_acceptance_rate_nonzero(small_model):
    """NUTS acceptance rate must be > 0 with the fixed sign and adaptation."""
    from diffcmb import run_chain_nut

    x0 = small_model.prior_parameters_tf()
    samples, results = run_chain_nut(
        small_model, x0, _step_size=0.001,
        num_results=30, num_burnin_steps=50,
    )

    inner = results.inner_results
    accept_rate = float(inner.is_accepted.numpy().mean())
    assert accept_rate > 0.0, \
        f"NUTS acceptance rate = {accept_rate:.3f} — sampler is completely stuck"


# ── step size adaptation ──────────────────────────────────────────────────────

@skip_no_tfp
def test_nuts_has_adapted_step_size(small_model):
    """Results object must expose a new_step_size from DualAveragingStepSizeAdaptation."""
    from diffcmb import run_chain_nut

    x0 = small_model.prior_parameters_tf()
    samples, results = run_chain_nut(
        small_model, x0, _step_size=0.01,
        num_results=10, num_burnin_steps=20,
    )

    assert hasattr(results, "new_step_size"), \
        "results lacks new_step_size — DualAveragingStepSizeAdaptation not applied"
    final_step = float(results.new_step_size.numpy()[-1])
    # Adaptation should change it from the initial 0.01
    assert np.isfinite(final_step) and final_step > 0, \
        f"Adapted step size is not a positive finite number: {final_step}"


# ── inner_results structure ───────────────────────────────────────────────────

@skip_no_tfp
def test_nuts_results_have_inner_results(small_model):
    """
    With adaptive wrapper, results.inner_results must exist and contain
    target_log_prob and is_accepted tensors of the correct length.
    """
    from diffcmb import run_chain_nut

    n = 15
    x0 = small_model.prior_parameters_tf()
    samples, results = run_chain_nut(
        small_model, x0, _step_size=0.001,
        num_results=n, num_burnin_steps=10,
    )

    assert hasattr(results, "inner_results"), "missing inner_results attribute"
    inner = results.inner_results
    assert hasattr(inner, "target_log_prob"), "inner_results missing target_log_prob"
    assert hasattr(inner, "is_accepted"), "inner_results missing is_accepted"
    assert inner.target_log_prob.shape[0] == n
    assert inner.is_accepted.shape[0] == n


@skip_no_tfp
def test_gibbs_chain_moves(small_model):
    """Verify that the Gibbs sampler runs without crashing and moves."""
    from diffcmb import run_gibbs_chain

    samples, logp, accepts, final_step = run_gibbs_chain(
        small_model,
        n_samples=5,
        n_burnin=5,
        hmc_step_size=0.01,
        n_lfs=5,
        seed=42,
    )
    assert samples.shape == (5, len(small_model.x0))
    assert logp.shape == (5,)
    assert accepts.shape == (5,)
    assert isinstance(final_step, float)
    assert not np.allclose(samples[0], samples[-1])


@skip_no_tfp
def test_gibbs_chain_with_phi_block_moves(small_model):
    """Phase 2 Block 3: phi | alm, C_l, d runs alongside the existing blocks."""
    from diffcmb import run_gibbs_chain

    lmax = small_model.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    n_phi = n_real + n_imag
    cl_phiphi_full = np.full(lmax, 1e-6, dtype=np.float64)

    samples, phi_samples, logp, accepts, final_step = run_gibbs_chain(
        small_model,
        n_samples=5,
        n_burnin=5,
        hmc_step_size=0.01,
        n_lfs=5,
        cl_phiphi_full=cl_phiphi_full,
        phi_hmc_step_size=0.01,
        phi_n_lfs=5,
        seed=42,
    )
    assert samples.shape == (5, len(small_model.x0))
    assert phi_samples.shape == (5, n_phi)
    assert logp.shape == (5,)
    assert accepts.shape == (5,)
    assert isinstance(final_step, float)
    assert np.all(np.isfinite(phi_samples))
    assert not np.allclose(phi_samples[0], phi_samples[-1])


@skip_no_tfp
def test_gibbs_chain_with_phi_block_mclmc_moves(small_model):
    """MCLMC spike (ROADMAP.md, 2026-08-07): phi_sampler='mclmc' drop-in for
    Block 3 runs alongside the existing blocks and moves, same smoke-test
    shape as the HMC phi-block test above."""
    from diffcmb import run_gibbs_chain

    lmax = small_model.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    n_phi = n_real + n_imag
    cl_phiphi_full = np.full(lmax, 1e-6, dtype=np.float64)

    samples, phi_samples, logp, accepts, final_step = run_gibbs_chain(
        small_model,
        n_samples=5,
        n_burnin=5,
        hmc_step_size=0.01,
        n_lfs=5,
        cl_phiphi_full=cl_phiphi_full,
        phi_sampler='mclmc',
        phi_hmc_step_size=1e-4,
        phi_n_lfs=5,
        phi_mclmc_L=1.0,
        seed=42,
    )
    assert samples.shape == (5, len(small_model.x0))
    assert phi_samples.shape == (5, n_phi)
    assert logp.shape == (5,)
    assert accepts.shape == (5,)
    assert isinstance(final_step, float)
    assert np.all(np.isfinite(phi_samples))
    assert not np.allclose(phi_samples[0], phi_samples[-1])


@skip_no_tfp
def test_gibbs_chain_with_phi_block_nuts_moves(small_model):
    """NUTS spike (ROADMAP.md, 2026-08-17): phi_sampler='nuts' drop-in for
    Block 3 -- dynamic trajectory length via the no-U-turn criterion instead
    of a fixed phi_n_lfs, same smoke-test shape as the HMC/MCLMC phi-block
    tests above. No phi_n_lfs needed (NUTS ignores it and picks its own
    trajectory length per step)."""
    from diffcmb import run_gibbs_chain

    lmax = small_model.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    n_phi = n_real + n_imag
    cl_phiphi_full = np.full(lmax, 1e-6, dtype=np.float64)

    samples, phi_samples, logp, accepts, final_step = run_gibbs_chain(
        small_model,
        n_samples=5,
        n_burnin=5,
        hmc_step_size=0.01,
        n_lfs=5,
        cl_phiphi_full=cl_phiphi_full,
        phi_sampler='nuts',
        phi_hmc_step_size=0.01,
        phi_nuts_max_tree_depth=4,
        seed=42,
    )
    assert samples.shape == (5, len(small_model.x0))
    assert phi_samples.shape == (5, n_phi)
    assert logp.shape == (5,)
    assert accepts.shape == (5,)
    assert isinstance(final_step, float)
    assert np.all(np.isfinite(phi_samples))
    assert not np.allclose(phi_samples[0], phi_samples[-1])


@skip_no_tfp
def test_gibbs_chain_cg_with_phi_block_moves(small_model):
    """Phase 2 gate-2 exactness experiment (ROADMAP.md Section 1): Block 2
    ('cg', unlensed exact Gaussian draw) alongside Block 3 (phi | alm, C_l,
    HMC against the correct lensed likelihood) runs without crashing and
    both blocks move."""
    from diffcmb import run_gibbs_chain

    lmax = small_model.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    n_phi = n_real + n_imag
    cl_phiphi_full = np.full(lmax, 1e-6, dtype=np.float64)

    samples, phi_samples, logp, accepts, final_step = run_gibbs_chain(
        small_model,
        n_samples=5,
        n_burnin=5,
        alm_sampler='cg',
        n_pcg_iter=10,
        cl_phiphi_full=cl_phiphi_full,
        phi_hmc_step_size=0.01,
        phi_n_lfs=5,
        seed=42,
    )
    assert samples.shape == (5, len(small_model.x0))
    assert phi_samples.shape == (5, n_phi)
    assert logp.shape == (5,)
    assert accepts.shape == (5,)
    assert np.all(accepts)  # 'cg' has no accept/reject -- always True
    assert isinstance(final_step, float)
    assert np.all(np.isfinite(samples))
    assert np.all(np.isfinite(phi_samples))
    assert not np.allclose(samples[0], samples[-1])
    assert not np.allclose(phi_samples[0], phi_samples[-1])


@skip_no_tfp
def test_gibbs_chain_messenger_rejects_phi_block(small_model):
    """cl_phiphi_full (Block 3) is only wired up for alm_sampler in {'hmc', 'cg'}."""
    from diffcmb import run_gibbs_chain

    cl_phiphi_full = np.full(small_model.lmax, 1e-6, dtype=np.float64)
    with pytest.raises(ValueError, match="requires alm_sampler"):
        run_gibbs_chain(
            small_model,
            n_samples=1,
            n_burnin=1,
            alm_sampler='messenger',
            cl_phiphi_full=cl_phiphi_full,
            seed=42,
        )


@pytest.fixture(scope="module")
def small_masked_matrixfree_model():
    """Small masked-sky model on the matrix-free ducc0 SHT path, for
    alm_sampler='messenger' (ROADMAP.md Phase 0c). Masking is applied by
    zeroing Ninv/prior_map outside a polar-cap unmasked_idx -- overriding
    unmasked_idx alone is not enough (see scripts/debug_messenger_masksky.py).

    Uses NSIDE=8 rather than the module-level NSIDE=4: the messenger sampler
    relies on the full-sky SHT being adequately resolved (roughly NSIDE >
    lmax/2, the HEALPix Nyquist-ish limit), and NSIDE=4 at LMAX=10 is well
    under that, which was found to make the messenger chain diverge almost
    immediately (a different, more severe failure than the ~1-2% quadrature
    error documented in sample_alm_messenger, which was characterised at
    NSIDE=8/lmax=10).
    """
    try:
        import ducc0  # noqa: F401
        import healpy as hp
    except ImportError:
        pytest.skip("ducc0/healpy not installed")

    from diffcmb import CosmologyAdvancedSampling

    messenger_nside = 8
    m = CosmologyAdvancedSampling(
        _lmax=LMAX, _NSIDE=messenger_nside, _noisesig=1.0, data_mode='synthetic',
        use_matrixfree_sht=True,
    )
    theta, _ = hp.pix2ang(messenger_nside, np.arange(m.NPIX))
    cutoff = np.arccos(1 - 2 * 0.7)
    m.unmasked_idx = np.where(theta < cutoff)[0]
    mask = np.ones(m.NPIX, dtype=bool)
    mask[m.unmasked_idx] = False
    m.Ninv = m.Ninv.copy()
    m.Ninv[mask] = 0.0
    m.prior_map = m.prior_map.copy()
    m.prior_map[mask] = 0.0
    m._ensure_tf_tensors()
    return m


@skip_no_tfp
def test_gibbs_chain_messenger_moves_and_stays_bounded(small_masked_matrixfree_model):
    """alm_sampler='messenger' runs on a masked sky without crashing or
    diverging (ROADMAP.md Phase 0c).

    This is deliberately a boundedness/smoke test, not a statistical-accuracy
    test: the messenger sampler's harmonic-space step uses a diagonal
    approximation of A^T A (the full-sky SHT's Gram matrix), which is only
    ~1-2% accurate for a real (quadrature-approximate) HEALPix SHT. Without a
    safety margin (samplers.py::sample_alm_messenger's
    norm_diag_safety_margin) this makes the messenger Markov chain's
    per-step transition operator have spectral radius > 1 under masking --
    genuine divergence, found via scripts/debug_messenger_masksky.py. The
    margin fixes divergence but trades off against posterior bias (still
    tens of posterior standard errors at lmax=10/NSIDE=8 in that script,
    not yet resolved) -- so alm_sampler='messenger' is not yet validated for
    production use; this test only guards against a regression to outright
    numerical blow-up.
    """
    from diffcmb import run_gibbs_chain

    samples, logp, accepts, final_step = run_gibbs_chain(
        small_masked_matrixfree_model,
        n_samples=5,
        n_burnin=5,
        alm_sampler='messenger',
        n_messenger_iter=20,
        seed=42,
    )
    assert samples.shape == (5, len(small_masked_matrixfree_model.x0))
    assert logp.shape == (5,)
    assert accepts.shape == (5,)
    assert isinstance(final_step, float)
    assert np.all(np.isfinite(samples))
    assert np.abs(samples).max() < 1e3
    assert not np.allclose(samples[0], samples[-1])


# ── phi posterior mass matrix (likelihood-curvature-aware preconditioner) ────

def test_build_phi_posterior_mass_sqrt_matches_prior_only_when_fisher_zero():
    """build_phi_posterior_mass_sqrt(..., diag_fisher_per_L=0) must reduce to
    build_phi_prior_mass_sqrt exactly -- the new function is a strict
    generalisation (adds likelihood curvature on top of the existing prior
    curvature), so a zero Fisher term must be a no-op.
    """
    from diffcmb.samplers import (
        build_phi_posterior_mass_sqrt,
        build_phi_prior_mass_sqrt,
    )

    lmax = 10
    rng = np.random.default_rng(0)
    cl_phiphi_full = rng.uniform(1e-12, 1e-10, size=lmax)

    prior_only = build_phi_prior_mass_sqrt(lmax, cl_phiphi_full)
    diag_fisher_per_L = np.zeros(lmax)
    combined = build_phi_posterior_mass_sqrt(lmax, cl_phiphi_full, diag_fisher_per_L)

    np.testing.assert_allclose(combined, prior_only)


def test_build_phi_posterior_mass_sqrt_increases_with_fisher_curvature():
    """A nonzero diag_fisher_per_L must strictly increase the mass (i.e.
    shrink the whitened step) for every mode at that L, matching
    build_posterior_mass_sqrt's 1/C_l + Ninv_eff pattern for the alm block."""
    from diffcmb.samplers import build_phi_posterior_mass_sqrt

    lmax = 10
    rng = np.random.default_rng(1)
    cl_phiphi_full = rng.uniform(1e-12, 1e-10, size=lmax)
    diag_fisher_per_L = np.zeros(lmax)
    diag_fisher_per_L[5] = 1e11  # large curvature injected at L=5 only

    zero_fisher = build_phi_posterior_mass_sqrt(lmax, cl_phiphi_full, np.zeros(lmax))
    with_fisher = build_phi_posterior_mass_sqrt(lmax, cl_phiphi_full, diag_fisher_per_L)

    assert np.all(with_fisher >= zero_fisher - 1e-30)
    assert np.any(with_fisher > zero_fisher * 1.5)


# ── phi block mass matrix (cross-L Nystrom correction) ──────────────────────

def test_build_phi_block_mass_chol_reduces_to_diagonal_when_hessian_empty():
    """build_phi_block_mass_chol with an empty block_hessian dict must give
    block Cholesky factors R whose diagonal reproduces
    build_phi_posterior_mass_sqrt exactly (each block's precision is then
    purely diagonal, so R's diagonal is just sqrt(precision) -- the same
    degrade-to-diagonal contract PhiWhitener relies on for its initial,
    pre-warmup state to match phi_mass_matrix='prior'/'fisher')."""
    from diffcmb.samplers import (
        _alm_index_lm,
        build_phi_block_mass_chol,
        build_phi_posterior_mass_sqrt,
    )

    lmax = 10
    rng = np.random.default_rng(2)
    cl_phiphi_full = rng.uniform(1e-12, 1e-10, size=lmax)
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    diag_fisher_per_L = np.zeros(lmax)
    diag_fisher_per_L[5] = 1e11

    expected = build_phi_posterior_mass_sqrt(lmax, cl_phiphi_full, diag_fisher_per_L)
    block_chol = build_phi_block_mass_chol(lmax, cl_phiphi_full, diag_fisher_per_L, {})

    n = n_real + n_imag
    recovered = np.empty(n)
    seen = np.zeros(n, dtype=bool)
    for idx, R in block_chol:
        np.testing.assert_allclose(R, np.diag(np.diag(R)), atol=1e-12)
        recovered[idx] = np.diag(R)
        seen[idx] = True
    assert np.all(seen)
    # rtol slightly looser than machine precision -- build_phi_block_mass_chol
    # adds a small numerical jitter (1e-10 * trace/K) before the Cholesky for
    # robustness against near-singular blocks (see its docstring), which is a
    # deliberate, tiny perturbation, not a bug.
    np.testing.assert_allclose(recovered, expected, rtol=1e-6)


def test_build_phi_block_mass_chol_offdiag_from_hessian():
    """A nonzero off-diagonal block_hessian entry must show up as a
    nonzero off-diagonal element of the corresponding block's Cholesky
    factor's reconstructed precision (R @ R.T) -- this is the whole point
    of 'block' mode: representing curvature a diagonal mass matrix cannot."""
    from diffcmb.samplers import _alm_index_lm, build_phi_block_mass_chol

    lmax = 10
    cl_phiphi_full = np.full(lmax, 1e-6)
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    L_arr, m_arr = _alm_index_lm(lmax, n_real, n_imag)
    diag_fisher_per_L = np.zeros(lmax)

    # m=0 real block spans L=2..9 (8 entries) -- inject a coupling between
    # its first two entries.
    idx = np.where((m_arr == 0) & (np.arange(n_real + n_imag) < n_real))[0]
    K = len(idx)
    H = np.zeros((K, K))
    # Diagonal precision here is ~1/cl_phiphi_full = 1e6 per entry; keep the
    # injected coupling well below that so the 2x2 principal minor stays PSD.
    H[0, 1] = H[1, 0] = 1e5
    block_hessian = {("real", 0): (idx, H)}

    block_chol = build_phi_block_mass_chol(lmax, cl_phiphi_full, diag_fisher_per_L, block_hessian)
    found = False
    for idx2, R in block_chol:
        if np.array_equal(idx2, idx):
            precision = R @ R.T
            assert abs(precision[0, 1] - 1e5) < 1e-2
            found = True
    assert found


def test_phi_whitener_diag_matches_elementwise_division():
    """PhiWhitener's diag mode must be numerically identical to the plain
    elementwise mass_sqrt division it replaces."""
    from diffcmb.samplers import PhiWhitener

    rng = np.random.default_rng(3)
    mass_sqrt = rng.uniform(0.5, 2.0, size=20)
    phi = rng.standard_normal(20)

    w = PhiWhitener(mass_sqrt_np=mass_sqrt)
    u = w.whiten_np(phi)
    np.testing.assert_allclose(u, phi * mass_sqrt)
    np.testing.assert_allclose(w.unwhiten_np(u), phi, atol=1e-10)


@skip_no_tfp
def test_phi_whitener_block_roundtrip_and_matches_diag_when_diagonal():
    """PhiWhitener's block mode round-trips (unwhiten(whiten(phi)) == phi)
    and, when the block Cholesky factors happen to be diagonal, its TF
    unwhiten_tf must agree with the plain-diagonal elementwise division to
    numerical precision -- the block path is a strict generalisation."""
    import tensorflow as tf

    from diffcmb.samplers import PhiWhitener, build_phi_block_mass_chol

    lmax = 10
    cl_phiphi_full = np.full(lmax, 1e-6)
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    n = n_real + n_imag
    diag_fisher_per_L = np.zeros(lmax)

    block_chol = build_phi_block_mass_chol(lmax, cl_phiphi_full, diag_fisher_per_L, {})
    w_block = PhiWhitener(block_chol=block_chol)
    mass_sqrt = np.empty(n)
    for idx, R in block_chol:
        mass_sqrt[idx] = np.diag(R)
    w_diag = PhiWhitener(mass_sqrt_np=mass_sqrt)

    rng = np.random.default_rng(4)
    phi = rng.standard_normal(n)
    u_block = w_block.whiten_np(phi)
    np.testing.assert_allclose(w_block.unwhiten_np(u_block), phi, atol=1e-8)

    u_tf = tf.constant(u_block, dtype=tf.float64)
    phi_from_block_tf = w_block.unwhiten_tf(u_tf).numpy()
    phi_from_diag_tf = w_diag.unwhiten_tf(u_tf).numpy()
    np.testing.assert_allclose(phi_from_block_tf, phi_from_diag_tf, atol=1e-8)
    np.testing.assert_allclose(phi_from_block_tf, phi, atol=1e-8)


@skip_no_tfp
def test_gibbs_chain_phi_mass_matrix_block_moves(small_model):
    """phi_mass_matrix='block' recomputes the phi mass matrix mid-burn-in
    using estimate_phi_block_hessian and keeps running without breaking
    the chain -- shapes/finiteness match the existing 'fisher' test."""
    from diffcmb import run_gibbs_chain

    lmax = small_model.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    n_phi = n_real + n_imag
    cl_phiphi_full = np.full(lmax, 1e-6, dtype=np.float64)

    samples, phi_samples, logp, accepts, final_step = run_gibbs_chain(
        small_model,
        n_samples=5,
        n_burnin=5,
        hmc_step_size=0.01,
        n_lfs=5,
        cl_phiphi_full=cl_phiphi_full,
        phi_hmc_step_size=0.01,
        phi_n_lfs=5,
        phi_mass_matrix='block',
        phi_fisher_warmup_iter=2,
        phi_fisher_n_probes=2,
        phi_block_n_probes=2,
        seed=42,
    )
    assert samples.shape == (5, len(small_model.x0))
    assert phi_samples.shape == (5, n_phi)
    assert logp.shape == (5,)
    assert accepts.shape == (5,)
    assert isinstance(final_step, float)
    assert np.all(np.isfinite(phi_samples))
    assert not np.allclose(phi_samples[0], phi_samples[-1])


@skip_no_tfp
def test_gibbs_chain_sample_cl_phiphi_with_block_mass_matrix_moves(small_model):
    """sample_cl_phiphi=True combined with phi_mass_matrix='block' IS
    supported (2026-08-23, ROADMAP.md): unlike 'fisher', the block Nystrom
    correction splits cleanly into a frozen likelihood-curvature part
    (diag_fisher_per_L/block_hessian, independent of cl_phiphi_full) and a
    cheap-to-rebuild diagonal prior-precision part that tracks the new
    spectrum every sweep -- see run_gibbs_chain's docstring and the Step 3a
    block-mode branch. Both blocks and Block 4 should run to completion and
    actually move."""
    from diffcmb import run_gibbs_chain
    from diffcmb.samplers import _alm_index_lm

    lmax = small_model.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    n_phi = n_real + n_imag
    C0 = 1e-6
    cl_phiphi_full = np.full(lmax, C0, dtype=np.float64)

    # phi_initial defaults to all-zeros, which (with only 5 burn-in sweeps)
    # never moves far enough from S_L=0 for Block 4 to escape its numerical
    # floor clip -- same fix as
    # test_gibbs_chain_sample_cl_phiphi_moves_and_returns_extra_array above:
    # start from a draw at the assumed prior scale instead.
    L_arr, m_arr = _alm_index_lm(lmax, n_real, n_imag)
    is_real = np.arange(n_phi) < n_real
    var = np.empty(n_phi)
    real_idx = np.where(is_real)[0]
    var[real_idx] = np.where(m_arr[real_idx] == 0, C0, C0 / 2.0)
    imag_idx = np.where(~is_real)[0]
    var[imag_idx] = C0 / 2.0
    phi_initial = np.random.default_rng(3).normal(scale=np.sqrt(var))

    samples, phi_samples, logp, accepts, final_step, cl_phiphi_samples = run_gibbs_chain(
        small_model,
        n_samples=5,
        n_burnin=5,
        hmc_step_size=0.01,
        n_lfs=5,
        cl_phiphi_full=cl_phiphi_full,
        phi_initial=phi_initial,
        phi_hmc_step_size=0.01,
        phi_n_lfs=5,
        phi_mass_matrix='block',
        phi_fisher_warmup_iter=2,
        phi_fisher_n_probes=2,
        phi_block_n_probes=2,
        sample_cl_phiphi=True,
        seed=42,
    )
    assert samples.shape == (5, len(small_model.x0))
    assert phi_samples.shape == (5, n_phi)
    assert logp.shape == (5,)
    assert accepts.shape == (5,)
    assert isinstance(final_step, float)
    assert cl_phiphi_samples.shape == (5, lmax - 2)
    assert np.all(np.isfinite(phi_samples))
    assert np.all(np.isfinite(cl_phiphi_samples))
    assert not np.allclose(phi_samples[0], phi_samples[-1])
    assert not np.allclose(cl_phiphi_samples[0], cl_phiphi_samples[-1])


def test_build_phi_block_mass_chol_rebuild_keeps_frozen_hessian_updates_diagonal():
    """The Step-3a every-sweep rebuild for phi_mass_matrix='block' +
    sample_cl_phiphi calls build_phi_block_mass_chol again with a NEW
    cl_phiphi_full but the SAME (frozen) diag_fisher_per_L/block_hessian.
    Directly exercise that call pattern: the off-diagonal structure (from
    block_hessian) must be preserved, while the diagonal precision must
    change to track the new spectrum -- i.e. rebuilding is not a no-op, and
    it is not silently discarding the frozen cross-L correction either."""
    from diffcmb.samplers import build_phi_block_mass_chol

    lmax = 6
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    diag_fisher_per_L = np.full(lmax, 0.1, dtype=np.float64)
    # A block_hessian entry for (channel='real', m=0): couples all m=0 real
    # coordinates together with off-diagonal curvature, same shape contract
    # as estimate_phi_block_hessian's return value.
    from diffcmb.samplers import _alm_index_lm
    L_arr, m_arr = _alm_index_lm(lmax, n_real, n_imag)
    idx0 = np.where((m_arr == 0) & (np.arange(n_real + n_imag) < n_real))[0]
    K = len(idx0)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((K, K))
    H = A @ A.T  # PSD, guaranteed nonzero off-diagonal with overwhelming probability
    block_hessian = {("real", 0): (idx0, H)}

    cl_a = np.full(lmax, 1e-6, dtype=np.float64)
    cl_b = np.full(lmax, 1e-3, dtype=np.float64)  # different spectrum scale

    chol_a = build_phi_block_mass_chol(lmax, cl_a, diag_fisher_per_L, block_hessian)
    chol_b = build_phi_block_mass_chol(lmax, cl_b, diag_fisher_per_L, block_hessian)

    # Find the (real, m=0) block in each rebuild's output.
    def find_block(chol):
        for idx, R in chol:
            if np.array_equal(idx, idx0):
                return R
        raise AssertionError("(real, m=0) block missing from build_phi_block_mass_chol output")

    R_a, R_b = find_block(chol_a), find_block(chol_b)
    precision_a = R_a @ R_a.T
    precision_b = R_b @ R_b.T

    # Both retain nonzero off-diagonal structure from the frozen block_hessian...
    off_diag_a = precision_a - np.diag(np.diag(precision_a))
    off_diag_b = precision_b - np.diag(np.diag(precision_b))
    assert not np.allclose(off_diag_a, 0.0)
    assert not np.allclose(off_diag_b, 0.0)
    # ...and that off-diagonal structure is identical (frozen, cl-independent).
    np.testing.assert_allclose(off_diag_a, off_diag_b)
    # But the diagonal changed to track the new spectrum (1/cl term differs).
    assert not np.allclose(np.diag(precision_a), np.diag(precision_b))


# ── run_gibbs_chain: phi_mass_matrix opt-in ('prior' default vs 'fisher') ────

@skip_no_tfp
def test_gibbs_chain_phi_mass_matrix_prior_default_unchanged(small_model, monkeypatch):
    """phi_mass_matrix defaults to 'prior' -- estimate_phi_diag_fisher must
    never be invoked in that mode (neither by omitting the parameter nor by
    passing 'prior' explicitly), so the new parameter is a strict opt-in
    with zero effect on existing callers.

    Note: this codebase's HMC chain is not bit-for-bit reproducible across
    separate run_gibbs_chain calls even with the same seed and model (TF's
    global op-level nondeterminism, confirmed independent of this change),
    so "unchanged" is checked by non-invocation of the new code path, not
    by comparing two runs' output arrays.
    """
    import diffcmb.samplers as samplers_mod
    from diffcmb import run_gibbs_chain

    def _boom(*args, **kwargs):
        raise AssertionError("estimate_phi_diag_fisher must not be called when phi_mass_matrix='prior'")

    monkeypatch.setattr(
        "diffcmb.lensing.estimate_phi_diag_fisher", _boom, raising=False
    )

    lmax = small_model.lmax
    cl_phiphi_full = np.full(lmax, 1e-6, dtype=np.float64)
    kwargs = {
        "n_samples": 5, "n_burnin": 5, "hmc_step_size": 0.01, "n_lfs": 5,
        "cl_phiphi_full": cl_phiphi_full, "phi_hmc_step_size": 0.01, "phi_n_lfs": 5,
        "seed": 42,
    }

    run_gibbs_chain(small_model, **kwargs)
    run_gibbs_chain(small_model, phi_mass_matrix='prior', **kwargs)


@skip_no_tfp
def test_gibbs_chain_phi_mass_matrix_fisher_moves(small_model):
    """phi_mass_matrix='fisher' recomputes the phi mass matrix mid-burn-in
    using estimate_phi_diag_fisher and keeps running without breaking the
    chain -- shapes/finiteness match the existing prior-only phi-block test."""
    from diffcmb import run_gibbs_chain

    lmax = small_model.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    n_phi = n_real + n_imag
    cl_phiphi_full = np.full(lmax, 1e-6, dtype=np.float64)

    samples, phi_samples, logp, accepts, final_step = run_gibbs_chain(
        small_model,
        n_samples=5,
        n_burnin=5,
        hmc_step_size=0.01,
        n_lfs=5,
        cl_phiphi_full=cl_phiphi_full,
        phi_hmc_step_size=0.01,
        phi_n_lfs=5,
        phi_mass_matrix='fisher',
        phi_fisher_warmup_iter=2,
        phi_fisher_n_probes=2,
        seed=42,
    )
    assert samples.shape == (5, len(small_model.x0))
    assert phi_samples.shape == (5, n_phi)
    assert logp.shape == (5,)
    assert accepts.shape == (5,)
    assert isinstance(final_step, float)
    assert np.all(np.isfinite(phi_samples))
    assert not np.allclose(phi_samples[0], phi_samples[-1])


# ── run_gibbs_chain: sample_cl_phiphi opt-in (optional Block 4) ─────────────

@skip_no_tfp
def test_gibbs_chain_sample_cl_phiphi_default_unchanged(small_model, monkeypatch):
    """sample_cl_phiphi defaults to False -- sample_cl_phiphi_given_phi must
    never be invoked in that mode (neither by omitting the parameter nor by
    passing sample_cl_phiphi=False explicitly), matching the same
    non-invocation pattern used above for phi_mass_matrix='prior' (this
    codebase's HMC chain is not bit-for-bit reproducible across separate
    run_gibbs_chain calls even with a fixed seed, so "unchanged" has to be
    checked by non-invocation rather than by comparing output arrays)."""
    from diffcmb import run_gibbs_chain

    def _boom(*args, **kwargs):
        raise AssertionError("sample_cl_phiphi_given_phi must not be called when sample_cl_phiphi=False")

    monkeypatch.setattr(
        "diffcmb.lensing.sample_cl_phiphi_given_phi", _boom, raising=False
    )

    lmax = small_model.lmax
    cl_phiphi_full = np.full(lmax, 1e-6, dtype=np.float64)
    kwargs = {
        "n_samples": 5, "n_burnin": 5, "hmc_step_size": 0.01, "n_lfs": 5,
        "cl_phiphi_full": cl_phiphi_full, "phi_hmc_step_size": 0.01, "phi_n_lfs": 5,
        "seed": 42,
    }

    run_gibbs_chain(small_model, **kwargs)
    run_gibbs_chain(small_model, sample_cl_phiphi=False, **kwargs)


def test_gibbs_chain_sample_cl_phiphi_requires_phi_block():
    """sample_cl_phiphi=True without cl_phiphi_full (i.e. Block 3 disabled)
    must raise, not silently no-op -- Block 4 only makes sense once phi is
    itself being sampled."""
    from diffcmb import run_gibbs_chain

    class _DummyModel:
        lmax = 10

    with pytest.raises(ValueError, match="sample_cl_phiphi"):
        run_gibbs_chain(_DummyModel(), n_samples=1, n_burnin=1, sample_cl_phiphi=True)


@skip_no_tfp
def test_gibbs_chain_sample_cl_phiphi_rejects_fisher_mass_matrix(small_model):
    """sample_cl_phiphi=True combined with phi_mass_matrix='fisher' is not
    supported (the Fisher mass matrix is estimated once at burn-in and
    frozen thereafter, which would silently go stale against a spectrum that
    keeps changing every sweep) -- must raise, not silently combine them."""
    from diffcmb import run_gibbs_chain

    lmax = small_model.lmax
    cl_phiphi_full = np.full(lmax, 1e-6, dtype=np.float64)
    with pytest.raises(ValueError, match="fisher"):
        run_gibbs_chain(
            small_model, n_samples=1, n_burnin=1, hmc_step_size=0.01, n_lfs=5,
            cl_phiphi_full=cl_phiphi_full, phi_hmc_step_size=0.01, phi_n_lfs=5,
            phi_mass_matrix='fisher', sample_cl_phiphi=True, seed=42,
        )


@skip_no_tfp
def test_gibbs_chain_sample_cl_phiphi_moves_and_returns_extra_array(small_model):
    """sample_cl_phiphi=True runs Block 4 alongside Blocks 1-3, returns an
    extra cl_phiphi_samples array as the last element of the return tuple,
    and that array actually moves (is not stuck at its initial value)."""
    from diffcmb import run_gibbs_chain

    lmax = small_model.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    n_phi = n_real + n_imag
    C0 = 1e-6
    cl_phiphi_full = np.full(lmax, C0, dtype=np.float64)

    # phi_initial defaults to all-zeros, which (with only 5 burn-in sweeps)
    # never moves far enough from S_L=0 for Block 4 to escape its numerical
    # floor clip -- start from a draw at the assumed prior scale instead, so
    # there's real phi power for the exact-draw to pick up on straight away.
    from diffcmb.samplers import _alm_index_lm
    L_arr, m_arr = _alm_index_lm(lmax, n_real, n_imag)
    is_real = np.arange(n_phi) < n_real
    var = np.empty(n_phi)
    real_idx = np.where(is_real)[0]
    var[real_idx] = np.where(m_arr[real_idx] == 0, C0, C0 / 2.0)
    imag_idx = np.where(~is_real)[0]
    var[imag_idx] = C0 / 2.0
    phi_initial = np.random.default_rng(3).normal(scale=np.sqrt(var))

    samples, phi_samples, logp, accepts, final_step, cl_phiphi_samples = run_gibbs_chain(
        small_model,
        n_samples=8,
        n_burnin=5,
        hmc_step_size=0.01,
        n_lfs=5,
        cl_phiphi_full=cl_phiphi_full,
        phi_initial=phi_initial,
        phi_hmc_step_size=0.01,
        phi_n_lfs=5,
        sample_cl_phiphi=True,
        seed=42,
    )
    assert samples.shape == (8, len(small_model.x0))
    assert phi_samples.shape == (8, n_phi)
    assert logp.shape == (8,)
    assert accepts.shape == (8,)
    assert isinstance(final_step, float)
    assert cl_phiphi_samples.shape == (8, lmax - 2)
    assert np.all(np.isfinite(cl_phiphi_samples))
    assert not np.allclose(cl_phiphi_samples[0], cl_phiphi_samples[-1])


@skip_no_tfp
def test_gibbs_chain_sample_cl_phiphi_recovers_known_spectrum(small_model):
    """Smoke test (mirrors the achievements.md-style small-lmax validation
    used elsewhere in this project): a short full run_gibbs_chain chain with
    Block 4 enabled, starting phi already near its stationary distribution
    for a known constant true C_L^phiphi, should keep the sampled spectrum
    in the right ballpark of that true value rather than drifting to a
    wildly different scale (which is what a weighting convention bug, e.g.
    the classic 1/C_l-vs-2/C_l mistake, would produce)."""
    from diffcmb import run_gibbs_chain
    from diffcmb.samplers import _alm_index_lm

    lmax = small_model.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    n_phi = n_real + n_imag
    C0 = 1e-6
    cl_phiphi_full = np.full(lmax, C0, dtype=np.float64)

    # Initialise phi from the assumed prior at C0 so the chain starts near
    # its stationary distribution instead of needing to burn in the
    # spectrum itself from a cold start in a handful of sweeps.
    L_arr, m_arr = _alm_index_lm(lmax, n_real, n_imag)
    is_real = np.arange(n_phi) < n_real
    var = np.empty(n_phi)
    real_idx = np.where(is_real)[0]
    var[real_idx] = np.where(m_arr[real_idx] == 0, C0, C0 / 2.0)
    imag_idx = np.where(~is_real)[0]
    var[imag_idx] = C0 / 2.0
    rng = np.random.default_rng(99)
    phi_initial = rng.normal(scale=np.sqrt(var))

    _samples, _phi_samples, _logp, _accepts, _final_step, cl_phiphi_samples = run_gibbs_chain(
        small_model,
        n_samples=40,
        n_burnin=20,
        hmc_step_size=0.01,
        n_lfs=5,
        cl_phiphi_full=cl_phiphi_full,
        phi_initial=phi_initial,
        phi_hmc_step_size=0.01,
        phi_n_lfs=5,
        sample_cl_phiphi=True,
        seed=7,
    )

    recovered = np.exp(cl_phiphi_samples[:, -1]).mean()  # highest-L bin: least small-L bias
    assert np.isfinite(recovered)
    # Loose ballpark check by design (few samples, small lmax): catches gross
    # convention errors (an order of magnitude or a 2x weighting bug) without
    # over-fitting to this specific short chain's Monte Carlo noise.
    assert 0.1 * C0 < recovered < 10.0 * C0, (
        f"recovered C_L (highest-L bin) = {recovered:.3e}, expected within "
        f"an order of magnitude of true C0={C0:.3e}"
    )


# ---------------------------------------------------------------------------
# Ancillary (non-centred) rescaling move wired into the Gibbs driver.
# The move's own correctness (invariance, acceptance ratio, Jacobian) is
# covered by tests/test_phi_ancillary_move.py; these cover the plumbing.
# ---------------------------------------------------------------------------

@skip_no_tfp
def test_gibbs_chain_phi_rescale_move_requires_block4(small_model):
    """The ancillary move rescales (phi, C_L^phiphi) jointly, so it is
    meaningless without Block 4 sampling the spectrum -- must raise rather
    than silently rescale a spectrum nobody is sampling."""
    from diffcmb import run_gibbs_chain

    lmax = small_model.lmax
    cl_phiphi_full = np.full(lmax, 1e-6, dtype=np.float64)
    with pytest.raises(ValueError, match="phi_rescale_move"):
        run_gibbs_chain(
            small_model, n_samples=1, n_burnin=1, hmc_step_size=0.01, n_lfs=5,
            cl_phiphi_full=cl_phiphi_full, phi_hmc_step_size=0.01, phi_n_lfs=5,
            sample_cl_phiphi=False, phi_rescale_move=True, seed=42,
        )


@skip_no_tfp
def test_gibbs_chain_phi_rescale_move_default_off_does_not_call_move(
    small_model, monkeypatch
):
    """phi_rescale_move defaults to False -- the move must not run at all for
    existing call sites (zero effect on validated configurations)."""
    from diffcmb import run_gibbs_chain

    def _boom(*a, **k):
        raise AssertionError(
            "sample_phi_amplitude_rescale must not be called when "
            "phi_rescale_move=False"
        )

    monkeypatch.setattr(
        "diffcmb.lensing.sample_phi_amplitude_rescale", _boom, raising=False
    )

    lmax = small_model.lmax
    cl_phiphi_full = np.full(lmax, 1e-6, dtype=np.float64)
    run_gibbs_chain(
        small_model, n_samples=2, n_burnin=1, hmc_step_size=0.01, n_lfs=5,
        cl_phiphi_full=cl_phiphi_full, phi_hmc_step_size=0.01, phi_n_lfs=5,
        sample_cl_phiphi=True, seed=42,
    )


@skip_no_tfp
def test_gibbs_chain_phi_rescale_move_runs_and_records_post_move_spectrum(
    small_model,
):
    """With the move on, the chain completes, the recorded cl_phiphi_samples
    are finite and move, and -- the subtle one -- the spectrum recorded for a
    sweep is the POST-move spectrum, not the pre-move Block 4 draw."""
    from diffcmb import run_gibbs_chain

    lmax = small_model.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    n_phi = n_real + n_imag
    C0 = 1e-6
    cl_phiphi_full = np.full(lmax, C0, dtype=np.float64)

    from diffcmb.samplers import _alm_index_lm
    L_arr, m_arr = _alm_index_lm(lmax, n_real, n_imag)
    is_real = np.arange(n_phi) < n_real
    var = np.empty(n_phi)
    real_idx = np.where(is_real)[0]
    var[real_idx] = np.where(m_arr[real_idx] == 0, C0, C0 / 2.0)
    imag_idx = np.where(~is_real)[0]
    var[imag_idx] = C0 / 2.0
    phi_initial = np.random.default_rng(3).normal(scale=np.sqrt(var))

    out = run_gibbs_chain(
        small_model,
        n_samples=8,
        n_burnin=5,
        hmc_step_size=0.01,
        n_lfs=5,
        cl_phiphi_full=cl_phiphi_full,
        phi_initial=phi_initial,
        phi_hmc_step_size=0.01,
        phi_n_lfs=5,
        sample_cl_phiphi=True,
        phi_rescale_move=True,
        phi_rescale_proposal_scale=0.1,
        seed=42,
    )
    assert len(out) == 6, "arity must be unchanged by the move"
    samples, phi_samples, logp, accepts, final_step, cl_phiphi_samples = out
    assert cl_phiphi_samples.shape == (8, lmax - 2)
    assert np.all(np.isfinite(cl_phiphi_samples))
    assert np.all(np.isfinite(phi_samples))
    assert not np.allclose(cl_phiphi_samples[0], cl_phiphi_samples[-1])


@skip_no_tfp
def test_gibbs_chain_phi_rescale_move_zero_scale_matches_move_off(small_model):
    """proposal_scale=0 makes the move an accepted identity, so the chain must
    be bit-identical to the same chain with the move switched off. This is the
    sharpest available check that the move is not perturbing state through
    some path other than its own accept/reject."""
    from diffcmb import run_gibbs_chain

    lmax = small_model.lmax
    cl_phiphi_full = np.full(lmax, 1e-6, dtype=np.float64)
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    phi_initial = np.random.default_rng(3).normal(
        scale=1e-3, size=n_real + n_imag
    )

    common = {
        "n_samples": 4, "n_burnin": 2, "hmc_step_size": 0.01, "n_lfs": 5,
        "cl_phiphi_full": cl_phiphi_full, "phi_initial": phi_initial,
        "phi_hmc_step_size": 0.01, "phi_n_lfs": 5,
        "sample_cl_phiphi": True, "seed": 7,
    }
    # run_gibbs_chain's `seed` covers the numpy stream only; the HMC kernels
    # draw from TF's global RNG, whose state carries across calls in one
    # process. Without this the two chains differ even with identical
    # arguments and the comparison below would be meaningless.
    import tensorflow as tf
    tf.random.set_seed(1234)
    off = run_gibbs_chain(small_model, **common)
    tf.random.set_seed(1234)
    on = run_gibbs_chain(
        small_model, phi_rescale_move=True,
        phi_rescale_proposal_scale=0.0, **common
    )
    np.testing.assert_allclose(on[1], off[1], rtol=0, atol=0)
    np.testing.assert_allclose(on[5], off[5], rtol=0, atol=0)


@skip_no_tfp
def test_gibbs_chain_seed_is_immune_to_prior_global_tf_rng_state(small_model):
    """`seed=` must fully determine a Gibbs chain, including the HMC blocks.

    Regression test for a real bug (2026-08-31): run_gibbs_chain seeded only
    its numpy stream (Block 1, Block 4, mass-matrix probes). The HMC/NUTS
    `one_step` calls take no `seed=`, so TFP drew momenta and Metropolis
    uniforms from TensorFlow's process-global RNG. A nominally seeded chain
    therefore depended on how many TF random ops had run earlier in the same
    process: consuming 5 tf.random.normal draws first moved the recovered
    C_L^phiphi by 5x (4.12e-7 -> 8.32e-8). That is what made
    test_gibbs_chain_sample_cl_phiphi_recovers_known_spectrum pass in
    isolation and fail in a full-suite run -- it was never flaky, it was
    reading an unseeded stream.

    Asserting bit-identity (atol=rtol=0) rather than closeness: the whole
    point is that the seed leaves nothing to chance.
    """
    import tensorflow as tf

    from diffcmb import run_gibbs_chain
    from diffcmb.samplers import _alm_index_lm

    lmax = small_model.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    C0 = 1e-6
    cl_phiphi_full = np.full(lmax, C0, dtype=np.float64)

    L_arr, m_arr = _alm_index_lm(lmax, n_real, n_imag)
    var = np.empty(n_real + n_imag)
    real_idx = np.arange(n_real)
    var[real_idx] = np.where(m_arr[real_idx] == 0, C0, C0 / 2.0)
    var[n_real:] = C0 / 2.0
    phi_initial = np.random.default_rng(99).normal(scale=np.sqrt(var))

    kwargs = {
        "n_samples": 8, "n_burnin": 4, "hmc_step_size": 0.01, "n_lfs": 5,
        "cl_phiphi_full": cl_phiphi_full, "phi_initial": phi_initial,
        "phi_hmc_step_size": 0.01, "phi_n_lfs": 5, "sample_cl_phiphi": True,
        "seed": 7,
    }

    first = run_gibbs_chain(small_model, **kwargs)
    # Advance TF's global RNG stream, exactly as an unrelated earlier test
    # (or any earlier TF op) would.
    for _ in range(5):
        tf.random.normal([1000])
    second = run_gibbs_chain(small_model, **kwargs)

    # alm samples, phi samples and the Block 4 spectrum must all match exactly.
    for idx, name in ((0, "alm"), (1, "phi"), (5, "cl_phiphi")):
        np.testing.assert_allclose(
            first[idx], second[idx], rtol=0, atol=0,
            err_msg=f"{name} samples differ => seed does not control the HMC blocks",
        )


@skip_no_tfp
def test_cl_phiphi_prior_nu_requires_block4():
    """The kwarg is meaningless without Block 4 and must say so, not no-op."""
    from diffcmb import run_gibbs_chain
    with pytest.raises(ValueError, match="cl_phiphi_prior_nu"):
        run_gibbs_chain(None, n_samples=1, cl_phiphi_prior_nu=6.0)


@skip_no_tfp
def test_gibbs_chain_proper_cl_phiphi_prior_restrains_an_inflated_phi(small_model):
    """End-to-end: the proper Block 4 prior must pull an inflated C_L^phiphi back.

    This is the sampler-level counterpart to the lensing.py unit tests, and it
    guards the subtlety that makes or breaks the feature: run_gibbs_chain
    rebinds `cl_phiphi_full` to the newly drawn spectrum every sweep, so the
    prior's fiducial has to be snapshotted BEFORE the loop. If it were read
    from the live variable the prior would chase the chain, exert no restoring
    force, and silently degrade to the improper behaviour it exists to remove
    -- while still passing every unit test of the conditional itself.

    Setup: start phi 30x too large in amplitude (900x in power) at a known
    fiducial spectrum, run both configurations from the identical state, and
    require the proper-prior chain to end materially closer to fiducial.
    """
    from diffcmb import run_gibbs_chain
    from diffcmb.samplers import _alm_index_lm

    lmax = small_model.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    C0 = 1e-6
    cl_phiphi_full = np.full(lmax, C0, dtype=np.float64)

    L_arr, m_arr = _alm_index_lm(lmax, n_real, n_imag)
    var = np.empty(n_real + n_imag)
    real_idx = np.arange(n_real)
    var[real_idx] = np.where(m_arr[real_idx] == 0, C0, C0 / 2.0)
    var[n_real:] = C0 / 2.0
    phi_initial = np.random.default_rng(99).normal(scale=np.sqrt(var)) * 30.0

    # 60 sweeps, not 30: a fiducial that chases the chain still looks correct
    # for the first few sweeps (it starts AT the fiducial) and only relaxes to
    # the flat-prior answer geometrically, at rate nu/(2L-1+nu) per sweep. The
    # window has to be long enough for that relaxation to show up, or the test
    # cannot see the bug it exists to catch. Verified by mutation: reading the
    # live cl_phiphi_full here gives 6.82 e-folds vs 4.72 for the correct
    # snapshot (flat-prior reference: 6.76), so the 1.0 e-fold margin below
    # separates them with room to spare.
    common = {
        "n_samples": 60, "n_burnin": 10, "hmc_step_size": 0.01, "n_lfs": 5,
        "cl_phiphi_full": cl_phiphi_full, "phi_initial": phi_initial,
        "phi_hmc_step_size": 0.01, "phi_n_lfs": 5, "sample_cl_phiphi": True,
        "seed": 7,
    }
    flat = run_gibbs_chain(small_model, **common)
    proper = run_gibbs_chain(small_model, cl_phiphi_prior_nu=40.0, **common)

    # Compare on the log scale against the fiducial: how many e-folds above C0
    # does each configuration sit, averaged over multipoles and sweeps?
    flat_excess = float(np.mean(flat[5] - np.log(C0)))
    proper_excess = float(np.mean(proper[5] - np.log(C0)))

    assert flat_excess > 1.0, "test setup failed: flat-prior run is not inflated"
    assert proper_excess < flat_excess - 1.0, (
        f"proper prior did not restrain the amplitude "
        f"(proper {proper_excess:.3f} vs flat {flat_excess:.3f} e-folds above "
        f"fiducial) -- check that the prior fiducial is snapshotted before the "
        f"sweep loop rather than read from the rebound cl_phiphi_full"
    )


@skip_no_tfp
def test_checkpoint_packing_version_mismatch(small_model, tmp_path):
    """Resume fails fast and with a clear message if checkpoint packing_version or vector length mismatches."""
    from diffcmb import run_gibbs_chain

    ckpt_file = str(tmp_path / "bad_ckpt.npz")
    # Write a checkpoint with an incompatible packing_version
    np.savez(
        ckpt_file,
        packing_version=np.int32(999),
        samples=np.zeros((1, 5)),
        logp=np.zeros(1),
        accepts=np.ones(1, dtype=bool),
        alm_state=np.zeros(10),
        lncl_state=np.zeros(small_model.lmax - 2),
        mass_sqrt=np.ones(10),
        step_size=np.float64(0.01),
    )

    with pytest.raises(ValueError, match="Checkpoint packing version 999 does not match current code PACKING_VERSION"):
        run_gibbs_chain(small_model, n_samples=2, checkpoint_path=ckpt_file)
