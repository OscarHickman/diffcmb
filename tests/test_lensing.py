"""Phase 1 lensing operator — gradient validation tests.

All tests require healpy (and most require TensorFlow).  They are automatically
skipped on environments where those libraries are absent (e.g. login nodes).

Validation strategy
-------------------
1. deflection_field: zero-phi → zero deflection; amplitude sanity check
2. precompute_lensing: weight normalisation
3. apply_lensing_tf: identity at zero phi; dL/dT_map autodiff vs FD
4. lens_map_phi_diff_tf: dL/dphi_alm autodiff vs FD  ← key Phase 1 check
5. psi_lensed: value matches unlensed posterior when phi=0 and noise→∞
6. alm end-to-end: dL/dalm through Y-matrix → apply_lensing pipeline vs FD
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

# Small problem size so tests run in seconds on a compute node
LMAX = 20
NSIDE = 16   # pixel size ≈ 220 arcmin; deflection ≪ pixel for typical phi amplitude


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_phi_packed(lmax, rng, amplitude=5e-4):
    """Random lensing potential in packed format; amplitude << pixel size."""
    from diffcmb.lensing import _alm_hp_to_packed
    size = hp.Alm.getsize(lmax)
    phi_hp = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    ells = np.array([hp.Alm.getlm(lmax, i)[0] for i in range(size)], dtype=float)
    ells = np.maximum(ells, 1.0)
    phi_hp *= amplitude / ells**1.5
    phi_hp[0] = 0.0   # monopole = 0
    if lmax >= 2:
        phi_hp[1] = 0.0   # l=1, m=0 = 0
    return _alm_hp_to_packed(phi_hp.astype(np.complex128), lmax)


def _rand_alm_packed(lmax, rng, scale=10.0):
    """Random CMB alm in packed (real+imag) format."""
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2
    return rng.standard_normal(n_real + n_imag).astype(np.float64) * scale


def _make_model(lmax=LMAX, nside=NSIDE):
    """Build a minimal synthetic model (no Planck data)."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from diffcmb import CosmologyAdvancedSampling
    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=100.0,
        data_mode="synthetic", dtype=tf.complex128
    )
    model._ensure_tf_tensors()
    return model


# ---------------------------------------------------------------------------
# 1 — deflection_field basics
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HEALPY, reason="healpy not installed")
def test_deflection_zero_phi():
    """Zero phi_alm → zero deflection."""
    from diffcmb.lensing import deflection_field
    phi_alm = np.zeros(hp.Alm.getsize(LMAX), dtype=complex)
    d_theta, d_phi = deflection_field(phi_alm, NSIDE, LMAX)
    assert np.allclose(d_theta, 0.0, atol=1e-14)
    assert np.allclose(d_phi, 0.0, atol=1e-14)


@pytest.mark.skipif(not HAS_HEALPY, reason="healpy not installed")
def test_deflection_amplitude_small():
    """Realistic phi gives deflection ≪ pixel size."""
    from diffcmb.lensing import _alm_packed_to_hp, deflection_field
    rng = np.random.default_rng(1)
    phi = _rand_phi_packed(LMAX, rng, amplitude=5e-4)
    phi_hp = _alm_packed_to_hp(phi, LMAX)
    d_theta, d_phi = deflection_field(phi_hp, NSIDE, LMAX)
    pixel_size_rad = np.pi / (4 * NSIDE)  # approximate
    assert np.max(np.abs(d_theta)) < pixel_size_rad
    assert np.max(np.abs(d_phi)) < pixel_size_rad


# ---------------------------------------------------------------------------
# 2 — format round-trip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HEALPY, reason="healpy not installed")
def test_alm_format_round_trip():
    """packed → hp → packed is lossless."""
    from diffcmb.lensing import _alm_hp_to_packed, _alm_packed_to_hp
    rng = np.random.default_rng(7)
    packed = _rand_phi_packed(LMAX, rng)
    hp_alm = _alm_packed_to_hp(packed, LMAX)
    packed2 = _alm_hp_to_packed(hp_alm, LMAX)
    np.testing.assert_allclose(packed, packed2, atol=1e-14)


# ---------------------------------------------------------------------------
# 3 — precompute_lensing
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HEALPY, reason="healpy not installed")
def test_precompute_weights_sum_to_one():
    """Bilinear weights always sum to 1 per pixel."""
    from diffcmb.lensing import _alm_packed_to_hp, precompute_lensing
    rng = np.random.default_rng(2)
    phi_hp = _alm_packed_to_hp(_rand_phi_packed(LMAX, rng), LMAX)
    pix = np.arange(hp.nside2npix(NSIDE))
    _, weights, _, _ = precompute_lensing(phi_hp, NSIDE, LMAX, pix)
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 4 — apply_lensing_tf: identity + dL/dT_map gradient
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_apply_lensing_identity_at_zero_phi():
    """Zero deflection: T_lensed == T_unlensed at all pixels."""
    from diffcmb.lensing import apply_lensing_tf, precompute_lensing
    npix = hp.nside2npix(NSIDE)
    phi_alm = np.zeros(hp.Alm.getsize(LMAX), dtype=complex)
    pix = np.arange(npix)
    neighbors, weights, _, _ = precompute_lensing(phi_alm, NSIDE, LMAX, pix)
    rng = np.random.default_rng(3)
    T = tf.constant(rng.standard_normal(npix), dtype=tf.float64)
    T_lensed = apply_lensing_tf(
        T,
        tf.constant(neighbors, tf.int32),
        tf.constant(weights, tf.float64),
    )
    np.testing.assert_allclose(T_lensed.numpy(), T.numpy(), atol=1e-10)


@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_apply_lensing_dT_grad_vs_fd():
    """dL/dT_map from TF autodiff agrees with finite differences."""
    from diffcmb.lensing import _alm_packed_to_hp, apply_lensing_tf, precompute_lensing
    npix = hp.nside2npix(NSIDE)
    rng = np.random.default_rng(17)
    phi_hp = _alm_packed_to_hp(_rand_phi_packed(LMAX, rng), LMAX)
    pix = np.arange(npix)
    neighbors, weights, _, _ = precompute_lensing(phi_hp, NSIDE, LMAX, pix)
    nbrs_tf = tf.constant(neighbors, tf.int32)
    wts_tf = tf.constant(weights, tf.float64)

    T_np = rng.standard_normal(npix)
    T_var = tf.Variable(T_np, dtype=tf.float64)

    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(apply_lensing_tf(T_var, nbrs_tf, wts_tf))
    g_auto = tape.gradient(loss, T_var).numpy()

    eps = 1e-5
    sampled = np.arange(0, npix, max(1, npix // 30))
    g_fd = np.zeros(npix)
    for i in sampled:
        T_p = T_np.copy()
        T_p[i] += eps
        T_m = T_np.copy()
        T_m[i] -= eps
        lp = tf.reduce_sum(apply_lensing_tf(tf.constant(T_p, tf.float64), nbrs_tf, wts_tf))
        lm = tf.reduce_sum(apply_lensing_tf(tf.constant(T_m, tf.float64), nbrs_tf, wts_tf))
        g_fd[i] = (lp.numpy() - lm.numpy()) / (2 * eps)

    np.testing.assert_allclose(
        g_auto[sampled], g_fd[sampled], rtol=1e-4, atol=1e-8,
        err_msg="dL/dT_map autodiff vs FD mismatch"
    )


# ---------------------------------------------------------------------------
# 5 — lens_map_phi_diff_tf: dL/dphi_alm gradient validation  ← KEY TEST
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_phi_grad_deflection_adjoint_vs_fd():
    """dL/dphi_alm from custom_gradient agrees with finite differences.

    Uses a simple loss = sum(T_lensed) and checks the packed phi gradient.
    This validates the full chain:
        phi_packed → deflection → bilinear weights → T_lensed
    and its reverse.
    """
    from diffcmb.lensing import lens_map_phi_diff_tf

    npix = hp.nside2npix(NSIDE)
    rng = np.random.default_rng(42)
    phi0 = _rand_phi_packed(LMAX, rng, amplitude=1e-4)
    T_np = rng.standard_normal(npix) * 50.0
    pix = np.arange(npix)

    phi_var = tf.Variable(phi0, dtype=tf.float64)
    T_tf = tf.constant(T_np, dtype=tf.float64)

    with tf.GradientTape() as tape:
        T_lensed = lens_map_phi_diff_tf(T_tf, phi_var, NSIDE, LMAX, pix)
        loss = tf.reduce_sum(T_lensed)
    g_auto = tape.gradient(loss, phi_var).numpy()

    # Finite differences over a subset of phi_packed components
    eps = 1e-6
    n_phi = len(phi0)
    sampled = np.arange(0, n_phi, max(1, n_phi // 20))
    g_fd = np.zeros(n_phi)
    for i in sampled:
        ph_p = phi0.copy()
        ph_p[i] += eps
        ph_m = phi0.copy()
        ph_m[i] -= eps
        lp = tf.reduce_sum(lens_map_phi_diff_tf(T_tf, tf.constant(ph_p, tf.float64), NSIDE, LMAX, pix))
        lm = tf.reduce_sum(lens_map_phi_diff_tf(T_tf, tf.constant(ph_m, tf.float64), NSIDE, LMAX, pix))
        g_fd[i] = (lp.numpy() - lm.numpy()) / (2 * eps)

    np.testing.assert_allclose(
        g_auto[sampled], g_fd[sampled], rtol=0.02, atol=1e-6,
        err_msg="dL/dphi_alm autodiff vs FD mismatch"
    )


# ---------------------------------------------------------------------------
# 6 — psi_lensed: value sanity + alm/phi gradient validation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_psi_lensed_zero_phi_matches_unlensed():
    """psi_lensed with phi=0 must equal model._psi_tf_raw (unlensed posterior)."""
    model = _make_model()
    lmax = model.lmax

    params_np = np.zeros(lmax - 2 + (lmax * (lmax + 1) // 2 - 3) + (lmax - 2) * (lmax - 1) // 2)
    params_np[: lmax - 2] = 5.0

    params_tf = tf.constant(params_np, dtype=tf.float64)
    n_phi = (lmax * (lmax + 1) // 2 - 3) + (lmax - 2) * (lmax - 1) // 2
    phi_tf = tf.zeros(n_phi, dtype=tf.float64)

    from diffcmb.lensing import psi_lensed
    psi_lens_val = psi_lensed(model, params_tf, phi_tf).numpy()
    psi_unlens_val = model._psi_tf_raw(params_tf).numpy()

    # With phi=0 the lensing is the identity so psi_lensed == _psi_tf_raw
    np.testing.assert_allclose(
        psi_lens_val, psi_unlens_val, rtol=1e-6,
        err_msg="psi_lensed(phi=0) ≠ _psi_tf_raw"
    )


@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_psi_lensed_alm_grad_vs_fd():
    """dL/dalm from TF autodiff on psi_lensed agrees with finite differences."""
    from diffcmb.lensing import psi_lensed
    model = _make_model()
    lmax = model.lmax
    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2

    rng = np.random.default_rng(11)
    params_np = np.zeros(n_lncl + n_real + n_imag)
    params_np[:n_lncl] = 5.0   # log C_l
    params_np[n_lncl:] = rng.standard_normal(n_real + n_imag) * 0.1
    phi_np = _rand_phi_packed(lmax, rng, amplitude=1e-4)

    params_var = tf.Variable(params_np, dtype=tf.float64)
    phi_tf = tf.constant(phi_np, dtype=tf.float64)

    with tf.GradientTape() as tape:
        val = psi_lensed(model, params_var, phi_tf)
    g_auto = tape.gradient(val, params_var).numpy()

    # FD on alm components only (skip lncl for speed)
    eps = 1e-5
    alm_slice = slice(n_lncl, n_lncl + 5)   # check first 5 alm coefficients
    g_fd = np.zeros(len(params_np))
    for i in range(n_lncl, n_lncl + 5):
        p_p = params_np.copy()
        p_p[i] += eps
        p_m = params_np.copy()
        p_m[i] -= eps
        lp = psi_lensed(model, tf.constant(p_p, tf.float64), phi_tf).numpy()
        lm = psi_lensed(model, tf.constant(p_m, tf.float64), phi_tf).numpy()
        g_fd[i] = (lp - lm) / (2 * eps)

    np.testing.assert_allclose(
        g_auto[alm_slice], g_fd[alm_slice], rtol=1e-4, atol=1e-6,
        err_msg="dL/dalm autodiff vs FD mismatch in psi_lensed"
    )


@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_psi_lensed_phi_grad_vs_fd():
    """dL/dphi_alm from TF autodiff on psi_lensed agrees with finite differences."""
    from diffcmb.lensing import psi_lensed
    model = _make_model()
    lmax = model.lmax
    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2
    n_phi = n_real + n_imag

    rng = np.random.default_rng(99)
    params_np = np.zeros(n_lncl + n_real + n_imag)
    params_np[:n_lncl] = 5.0
    params_np[n_lncl:] = rng.standard_normal(n_real + n_imag) * 0.1
    phi_np = _rand_phi_packed(lmax, rng, amplitude=1e-4)

    params_tf = tf.constant(params_np, dtype=tf.float64)
    phi_var = tf.Variable(phi_np, dtype=tf.float64)

    with tf.GradientTape() as tape:
        val = psi_lensed(model, params_tf, phi_var)
    g_auto = tape.gradient(val, phi_var).numpy()

    # FD on first 8 phi components
    eps = 1e-6
    g_fd = np.zeros(n_phi)
    for i in range(min(8, n_phi)):
        ph_p = phi_np.copy()
        ph_p[i] += eps
        ph_m = phi_np.copy()
        ph_m[i] -= eps
        lp = psi_lensed(model, params_tf, tf.constant(ph_p, tf.float64)).numpy()
        lm = psi_lensed(model, params_tf, tf.constant(ph_m, tf.float64)).numpy()
        g_fd[i] = (lp - lm) / (2 * eps)

    np.testing.assert_allclose(
        g_auto[:8], g_fd[:8], rtol=0.02, atol=1e-5,
        err_msg="dL/dphi_alm autodiff vs FD mismatch in psi_lensed"
    )
