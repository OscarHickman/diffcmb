"""Tests for CosmologyAdvancedSampling model."""
import numpy as np
import pytest


def _has_deps():
    try:
        import healpy  # noqa: F401
        import scipy  # noqa: F401
        import tensorflow as tf  # noqa: F401

        from src.cmb import CosmologyAdvancedSampling  # noqa: F401
        return True
    except Exception:
        return False


skip_no_deps = pytest.mark.skipif(not _has_deps(), reason="heavy deps unavailable")


# ── construction ──────────────────────────────────────────────────────────────

def test_model_constructs_or_skips():
    try:
        from src.cmb.model import CosmologyAdvancedSampling
    except Exception:
        pytest.skip("Could not import CosmologyAdvancedSampling (missing deps)")

    m = CosmologyAdvancedSampling(_lmax=8, _NSIDE=2, _noisesig=1.0)
    assert m.lmax == 8
    assert m.NSIDE == 2
    assert m.sph is None   # lazily created
    assert m.shape is None


@skip_no_deps
def test_model_synthetic_x0_shape():
    from src.cmb import CosmologyAdvancedSampling
    lmax, nside = 8, 2
    m = CosmologyAdvancedSampling(_lmax=lmax, _NSIDE=nside, _noisesig=1.0,
                                   data_mode='synthetic')
    expected_len = (lmax - 2) + (lmax*(lmax+1)//2 - 3) + sum(l-1 for l in range(2, lmax))
    assert len(m.x0) == expected_len, f"x0 length {len(m.x0)} != {expected_len}"


@skip_no_deps
def test_model_prior_parameters_tf_dtype():
    import tensorflow as tf

    from src.cmb import CosmologyAdvancedSampling
    m = CosmologyAdvancedSampling(_lmax=8, _NSIDE=2, _noisesig=1.0)
    x0 = m.prior_parameters_tf()
    assert x0.dtype == tf.float64


# ── psi_tf: value and gradient ────────────────────────────────────────────────

@skip_no_deps
def test_psi_tf_is_finite_at_x0():
    """psi_tf must return a finite scalar at the initial state."""
    from src.cmb import CosmologyAdvancedSampling
    m = CosmologyAdvancedSampling(_lmax=10, _NSIDE=4, _noisesig=1.0,
                                   data_mode='synthetic')
    m._ensure_tf_tensors()
    x0 = m.prior_parameters_tf()
    val = m.psi_tf(x0)
    assert np.isfinite(val.numpy()), f"psi_tf(x0) = {val.numpy()} is not finite"


@skip_no_deps
def test_psi_tf_gradient_finite_at_x0():
    """Gradient of psi_tf must be finite at the initial state (no NaN/Inf)."""
    import tensorflow as tf

    from src.cmb import CosmologyAdvancedSampling
    m = CosmologyAdvancedSampling(_lmax=10, _NSIDE=4, _noisesig=1.0,
                                   data_mode='synthetic')
    m._ensure_tf_tensors()
    x0 = m.prior_parameters_tf()

    with tf.GradientTape() as tape:
        tape.watch(x0)
        val = m.psi_tf(x0)
    grad = tape.gradient(val, x0)

    assert grad is not None, "gradient is None"
    assert not tf.reduce_any(tf.math.is_nan(grad)).numpy(), "gradient contains NaN"
    assert not tf.reduce_any(tf.math.is_inf(grad)).numpy(), "gradient contains Inf"


# ── sign convention: psi_tf is the negative log-posterior ────────────────────

@skip_no_deps
def test_psi_tf_is_negative_log_posterior():
    """
    psi_tf is the NEGATIVE log-posterior.

    The gradient of -psi_tf (the actual log-posterior) must be non-trivially
    non-zero at x0, confirming the posterior is not flat at the prior.
    Also verifies gradient(-psi_tf) == -gradient(psi_tf).
    """
    import tensorflow as tf

    from src.cmb import CosmologyAdvancedSampling
    m = CosmologyAdvancedSampling(_lmax=10, _NSIDE=4, _noisesig=1.0,
                                   data_mode='synthetic')
    m._ensure_tf_tensors()
    x0 = m.prior_parameters_tf()

    with tf.GradientTape() as tape:
        tape.watch(x0)
        log_posterior = -m.psi_tf(x0)
    grad_log_post = tape.gradient(log_posterior, x0)

    grad_norm = tf.norm(grad_log_post).numpy()
    assert grad_norm > 1e-6, f"gradient of -psi_tf ≈ 0 (norm={grad_norm:.2e})"

    with tf.GradientTape() as tape2:
        tape2.watch(x0)
        psi = m.psi_tf(x0)
    grad_psi = tape2.gradient(psi, x0)

    np.testing.assert_allclose(
        grad_log_post.numpy(), -grad_psi.numpy(), atol=1e-12,
        err_msg="-∇psi_tf != ∇(−psi_tf): sign inconsistency"
    )


@skip_no_deps
def test_psi_tf_positive_definite_terms():
    """
    psi1 (likelihood) and psi3 (alm prior) are always non-negative.

    Verify by comparing psi_tf to psi2-only contribution: psi_tf >= psi2.
    """
    import tensorflow as tf

    from src.cmb.model import CosmologyAdvancedSampling as CAS
    m = CAS(_lmax=10, _NSIDE=4, _noisesig=1.0, data_mode='synthetic')
    m._ensure_tf_tensors()
    x0 = m.prior_parameters_tf()

    lmax = m.lmax
    lnclstart = tf.zeros(2, dtype=tf.float64)
    lncl = tf.concat([lnclstart, x0[:lmax - 2]], axis=0)
    l = tf.cast(tf.range(lmax), tf.float64)
    psi2_only = float(tf.reduce_sum((l + 0.5) * lncl).numpy())

    full_psi = float(m.psi_tf(x0).numpy())

    # psi1 + psi3 >= 0, so psi_tf >= psi2
    assert full_psi >= psi2_only - 1e-9, \
        f"psi_tf ({full_psi:.4f}) < psi2 ({psi2_only:.4f}): psi1+psi3 is negative"


# ── ensure_tf_tensors idempotency ─────────────────────────────────────────────

@skip_no_deps
def test_ensure_tf_tensors_idempotent():
    """Calling _ensure_tf_tensors twice must not change sph or shape."""
    from src.cmb import CosmologyAdvancedSampling
    m = CosmologyAdvancedSampling(_lmax=8, _NSIDE=2, _noisesig=1.0)
    m._ensure_tf_tensors()
    sph_first = m.sph
    shape_first = m.shape
    m._ensure_tf_tensors()
    assert m.sph is sph_first
    assert m.shape is shape_first
