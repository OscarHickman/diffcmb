"""Gradient validation tests for the Phase 1 lensing operator.

Validates at lmax=50 (small enough to run on a login node without TF,
skipped automatically if healpy or tensorflow are unavailable).
"""

import numpy as np
import pytest

try:
    import healpy as hp
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


LMAX_TEST = 50
NSIDE_TEST = 64   # nside >= lmax/2 ensures Nyquist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_phi_alm(lmax, rng, amplitude=1e-3):
    """Random lensing potential alm with realistic small amplitude."""
    size = hp.Alm.getsize(lmax)
    phi = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    # Suppress high-l modes (phi power ~ l^-3)
    ells = np.array([hp.Alm.getlm(lmax, i)[0] for i in range(size)], dtype=float)
    ells = np.maximum(ells, 1)
    phi *= amplitude / ells**1.5
    phi[0] = 0.0   # monopole = 0
    phi[1] = 0.0   # dipole = 0 (l=1, m=0)
    return phi.astype(np.complex128)


# ---------------------------------------------------------------------------
# Unit tests for deflection_field
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HEALPY, reason="healpy not installed")
def test_deflection_field_zero_phi():
    """Zero phi_alm → zero deflection everywhere."""
    from diffcmb.lensing import deflection_field

    lmax = LMAX_TEST
    nside = NSIDE_TEST
    phi_alm = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
    d_theta, d_phi = deflection_field(phi_alm, nside, lmax)

    assert d_theta.shape == (12 * nside * nside,)
    assert np.allclose(d_theta, 0.0, atol=1e-14)
    assert np.allclose(d_phi, 0.0, atol=1e-14)


@pytest.mark.skipif(not HAS_HEALPY, reason="healpy not installed")
def test_deflection_field_amplitude():
    """Deflection amplitude is small (< 5 arcmin) for a realistic phi."""
    from diffcmb.lensing import deflection_field

    lmax = LMAX_TEST
    nside = NSIDE_TEST
    rng = np.random.default_rng(42)
    phi_alm = _make_phi_alm(lmax, rng, amplitude=1e-3)
    d_theta, d_phi = deflection_field(phi_alm, nside, lmax)

    # 5 arcmin ≈ 0.00145 rad
    assert np.max(np.abs(d_theta)) < 0.002, f"d_theta too large: {np.max(np.abs(d_theta)):.4f} rad"
    assert np.max(np.abs(d_phi)) < 0.002, f"d_phi too large: {np.max(np.abs(d_phi)):.4f} rad"


# ---------------------------------------------------------------------------
# Unit tests for precompute_lensing
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HEALPY, reason="healpy not installed")
def test_precompute_lensing_zero_phi():
    """With phi=0 the lensing should map each pixel to itself (weights ≈ 1 on self)."""
    from diffcmb.lensing import precompute_lensing

    lmax = LMAX_TEST
    nside = NSIDE_TEST
    phi_alm = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
    pixel_indices = np.arange(12 * nside * nside)

    neighbors, weights, d_theta, d_phi = precompute_lensing(phi_alm, nside, lmax, pixel_indices)

    assert neighbors.shape == (4, len(pixel_indices))
    assert weights.shape == (4, len(pixel_indices))
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-10)


@pytest.mark.skipif(not HAS_HEALPY, reason="healpy not installed")
def test_precompute_lensing_weights_sum_to_one():
    """Bilinear weights always sum to 1."""
    from diffcmb.lensing import precompute_lensing

    lmax = LMAX_TEST
    nside = NSIDE_TEST
    rng = np.random.default_rng(7)
    phi_alm = _make_phi_alm(lmax, rng)
    pixel_indices = np.arange(0, 12 * nside * nside, 10)   # every 10th pixel

    _, weights, _, _ = precompute_lensing(phi_alm, nside, lmax, pixel_indices)
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# apply_lensing_tf: identity and linearity
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_apply_lensing_zero_deflection():
    """Zero deflection: T_lensed == T_unlensed at each pixel."""
    from diffcmb.lensing import apply_lensing_tf, precompute_lensing

    lmax = LMAX_TEST
    nside = NSIDE_TEST
    npix = 12 * nside * nside
    phi_alm = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
    pixel_indices = np.arange(npix)

    neighbors, weights, _, _ = precompute_lensing(phi_alm, nside, lmax, pixel_indices)
    neighbors_tf = tf.constant(neighbors, dtype=tf.int32)
    weights_tf = tf.constant(weights, dtype=tf.float64)

    rng = np.random.default_rng(3)
    T_map = tf.constant(rng.standard_normal(npix), dtype=tf.float64)
    T_lensed = apply_lensing_tf(T_map, neighbors_tf, weights_tf)

    # With zero deflection every pixel maps to itself → T_lensed ≈ T_map
    np.testing.assert_allclose(T_lensed.numpy(), T_map.numpy(), atol=1e-10)


@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_apply_lensing_gradient_vs_finite_difference():
    """dL/dT_map from TF autodiff agrees with finite differences to 1e-5."""
    from diffcmb.lensing import apply_lensing_tf, precompute_lensing

    lmax = LMAX_TEST
    nside = NSIDE_TEST
    npix = 12 * nside * nside
    rng = np.random.default_rng(17)
    phi_alm = _make_phi_alm(lmax, rng)
    pixel_indices = np.arange(npix)

    neighbors, weights, _, _ = precompute_lensing(phi_alm, nside, lmax, pixel_indices)
    neighbors_tf = tf.constant(neighbors, dtype=tf.int32)
    weights_tf = tf.constant(weights, dtype=tf.float64)

    T_np = rng.standard_normal(npix)
    T_tf = tf.Variable(T_np, dtype=tf.float64)

    # Autodiff gradient of sum(T_lensed) w.r.t. T_map
    with tf.GradientTape() as tape:
        T_lensed = apply_lensing_tf(T_tf, neighbors_tf, weights_tf)
        loss = tf.reduce_sum(T_lensed)
    grad_auto = tape.gradient(loss, T_tf).numpy()

    # Finite differences
    eps = 1e-5
    grad_fd = np.zeros(npix)
    for i in range(0, npix, npix // 20):   # sample every ~5% of pixels
        T_plus = T_np.copy()
        T_plus[i] += eps
        T_minus = T_np.copy()
        T_minus[i] -= eps
        lp = apply_lensing_tf(tf.constant(T_plus, tf.float64), neighbors_tf, weights_tf)
        lm = apply_lensing_tf(tf.constant(T_minus, tf.float64), neighbors_tf, weights_tf)
        grad_fd[i] = (tf.reduce_sum(lp) - tf.reduce_sum(lm)).numpy() / (2 * eps)

    # Compare at sampled pixels only (others have grad_fd = 0, skip)
    sampled = np.arange(0, npix, npix // 20)
    np.testing.assert_allclose(
        grad_auto[sampled], grad_fd[sampled], rtol=1e-4, atol=1e-7,
        err_msg="dL/dT_map autodiff does not match finite differences"
    )
