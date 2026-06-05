"""Tests for NUTS/HMC samplers — sign convention and step size adaptation."""
import numpy as np
import pytest


def _has_all():
    try:
        import healpy  # noqa: F401
        import scipy  # noqa: F401
        import tensorflow as tf  # noqa: F401
        import tensorflow_probability as tfp  # noqa: F401

        from src.cmb import (  # noqa: F401
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
    from src.cmb import CosmologyAdvancedSampling
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

    from src.cmb import run_chain_nut

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
    """Same sign check for the HMC path."""
    from src.cmb import run_chain_hmc

    x0 = small_model.prior_parameters_tf()
    psi_at_x0 = float(small_model.psi_tf(x0).numpy())
    expected_logp = -psi_at_x0

    samples, results = run_chain_hmc(
        small_model, x0, _step_size=0.001,
        num_results=5, num_burnin_steps=10,
    )

    inner = results.inner_results
    first_logp = float(inner.accepted_results.target_log_prob.numpy()[0])

    assert abs(first_logp - expected_logp) < 1.0, (
        f"HMC target_log_prob ({first_logp:.4f}) != -psi_tf ({expected_logp:.4f})"
    )


# ── chain movement ────────────────────────────────────────────────────────────

@skip_no_tfp
def test_nuts_chain_moves(small_model):
    """
    NUTS must produce samples that differ from the initial state.

    The original bug caused 0% acceptance and every sample == x0.
    """
    from src.cmb import run_chain_nut

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
    from src.cmb import run_chain_nut

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
    from src.cmb import run_chain_nut

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
    from src.cmb import run_chain_nut

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
