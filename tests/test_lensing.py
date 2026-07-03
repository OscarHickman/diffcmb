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

    eps note (bug history, Phase 2 ROADMAP.md): a single phi_alm component
    perturbs the deflection field at every pixel simultaneously, and
    hp.get_interp_weights' bilinear scheme is only C0 (continuous) not C1
    (its derivative has genuine kinks at interpolation-cell boundaries).
    With eps=1e-6 (the original value here), a handful of the ~600 unmasked
    pixels' lensed positions happened to cross such a boundary within the
    perturbation, making *this FD reference* — not the analytic gradient —
    unstable: verified directly that this same FD estimate changes sign
    and swings by >100% between eps=1e-6 and eps=1e-7 for some components,
    while agreeing with the analytic gradient to ~1e-5 relative once eps is
    small enough (<=3e-9) not to cross a boundary. eps=1e-9 is safely in
    that regime (deflection_field is exactly linear in phi_alm, so no
    truncation/roundoff tradeoff pushes eps this small — checked stable
    across 1e-9..3e-10).
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
    eps = 1e-9
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


@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_lens_map_phi_diff_tf_traceable_in_tf_function():
    """lens_map_phi_diff_tf must survive tf.function tracing (Phase 1.5,
    ROADMAP.md): both the bilinear-geometry precompute and the FD backward
    pass now go through tf.py_function rather than a bare .numpy() call
    (mirroring sht_ducc.py's masked_synthesis_tf), so this op can sit inside
    samplers.py's @tf.function-decorated grad/matvec wrappers, not just run
    in pure eager mode. Also checks the forward value and both gradients
    (w.r.t. T_map and phi) match the eager-mode result exactly."""
    from diffcmb.lensing import lens_map_phi_diff_tf

    npix = hp.nside2npix(NSIDE)
    rng = np.random.default_rng(7)
    phi0 = _rand_phi_packed(LMAX, rng, amplitude=1e-4)
    T_np = rng.standard_normal(npix) * 50.0
    pix = np.arange(npix)

    T_tf = tf.constant(T_np, dtype=tf.float64)
    phi_tf = tf.constant(phi0, dtype=tf.float64)

    def _run(T, phi):
        with tf.GradientTape() as tape:
            tape.watch([T, phi])
            out = lens_map_phi_diff_tf(T, phi, NSIDE, LMAX, pix)
            loss = tf.reduce_sum(out ** 2)
        g_T, g_phi = tape.gradient(loss, [T, phi])
        return out, g_T, g_phi

    out_eager, gT_eager, gphi_eager = _run(T_tf, phi_tf)

    traced = tf.function(_run)
    out_traced, gT_traced, gphi_traced = traced(T_tf, phi_tf)

    np.testing.assert_allclose(out_traced.numpy(), out_eager.numpy())
    np.testing.assert_allclose(gT_traced.numpy(), gT_eager.numpy())
    np.testing.assert_allclose(gphi_traced.numpy(), gphi_eager.numpy())
    assert np.all(np.isfinite(gphi_traced.numpy()))


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
    """dL/dphi_alm from TF autodiff on psi_lensed agrees with finite differences.

    eps note: see test_phi_grad_deflection_adjoint_vs_fd — a coarse FD eps
    here perturbs every pixel's lensed position simultaneously and can cross
    a genuine (C0-but-not-C1) HEALPix bilinear-interpolation-cell boundary,
    making the FD reference itself unstable rather than the analytic
    gradient being wrong. eps=1e-9 avoids that (checked stable down to 3e-10;
    deflection_field is exactly linear in phi_alm so there's no
    truncation/roundoff tradeoff forcing eps larger).
    """
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
    eps = 1e-9
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


# ---------------------------------------------------------------------------
# 7 — log_prob_phi_block (Phase 2, Block 3 target) — value + gradient checks
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_log_prob_phi_block_zero_prior_matches_neg_psi_lensed():
    """With an infinite phi prior variance (cl_phiphi -> inf), the phi prior
    term vanishes and log_prob_phi_block reduces to -psi_lensed."""
    from diffcmb.lensing import log_prob_phi_block, psi_lensed
    model = _make_model()
    lmax = model.lmax
    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2

    rng = np.random.default_rng(7)
    params_np = np.zeros(n_lncl + n_real + n_imag)
    params_np[:n_lncl] = 5.0
    params_tf = tf.constant(params_np, dtype=tf.float64)
    phi_tf = tf.constant(_rand_phi_packed(lmax, rng, amplitude=1e-4), dtype=tf.float64)

    cl_phiphi_huge = np.full(lmax, 1e30)
    log_prob = log_prob_phi_block(model, params_tf, phi_tf, cl_phiphi_huge).numpy()
    neg_psi = -psi_lensed(model, params_tf, phi_tf).numpy()

    np.testing.assert_allclose(
        log_prob, neg_psi, rtol=1e-6,
        err_msg="log_prob_phi_block with cl_phiphi->inf should match -psi_lensed"
    )


@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_log_prob_phi_block_grad_vs_fd():
    """dlog_prob/dphi_alm from TF autodiff agrees with finite differences.

    This is the gradient Block 3's HMC step will need every leapfrog
    iteration, so it must be correct before Phase 2 wiring begins.
    """
    from diffcmb.lensing import log_prob_phi_block
    model = _make_model()
    lmax = model.lmax
    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2
    n_phi = n_real + n_imag

    rng = np.random.default_rng(13)
    params_np = np.zeros(n_lncl + n_real + n_imag)
    params_np[:n_lncl] = 5.0
    params_np[n_lncl:] = rng.standard_normal(n_real + n_imag) * 0.1
    params_tf = tf.constant(params_np, dtype=tf.float64)

    phi_np = _rand_phi_packed(lmax, rng, amplitude=1e-4)
    phi_var = tf.Variable(phi_np, dtype=tf.float64)
    cl_phiphi = np.full(lmax, 1e-8)   # tight but finite prior

    with tf.GradientTape() as tape:
        val = log_prob_phi_block(model, params_tf, phi_var, cl_phiphi)
    g_auto = tape.gradient(val, phi_var).numpy()

    eps = 1e-6
    g_fd = np.zeros(n_phi)
    for i in range(min(8, n_phi)):
        ph_p = phi_np.copy()
        ph_p[i] += eps
        ph_m = phi_np.copy()
        ph_m[i] -= eps
        lp = log_prob_phi_block(model, params_tf, tf.constant(ph_p, tf.float64), cl_phiphi).numpy()
        lm = log_prob_phi_block(model, params_tf, tf.constant(ph_m, tf.float64), cl_phiphi).numpy()
        g_fd[i] = (lp - lm) / (2 * eps)

    np.testing.assert_allclose(
        g_auto[:8], g_fd[:8], rtol=0.02, atol=1e-5,
        err_msg="dlog_prob/dphi_alm autodiff vs FD mismatch in log_prob_phi_block"
    )
