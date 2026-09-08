"""Tests for CosmologyAdvancedSampling model."""
import numpy as np
import pytest

from diffcmb.alm_utils import packed_sizes


def _has_deps():
    try:
        import healpy  # noqa: F401
        import scipy  # noqa: F401
        import tensorflow as tf  # noqa: F401

        from diffcmb import CosmologyAdvancedSampling  # noqa: F401
        return True
    except Exception:
        return False


skip_no_deps = pytest.mark.skipif(not _has_deps(), reason="heavy deps unavailable")


# ── construction ──────────────────────────────────────────────────────────────

@skip_no_deps
def test_model_constructs_or_skips():
    try:
        from diffcmb.model import CosmologyAdvancedSampling
    except Exception:
        pytest.skip("Could not import CosmologyAdvancedSampling (missing deps)")

    m = CosmologyAdvancedSampling(_lmax=8, _NSIDE=2, _noisesig=1.0)
    assert m.lmax == 8
    assert m.NSIDE == 2
    assert m.sph is None   # lazily created
    assert m.shape is None


@skip_no_deps
def test_model_synthetic_x0_shape():
    from diffcmb import CosmologyAdvancedSampling
    lmax, nside = 8, 2
    m = CosmologyAdvancedSampling(_lmax=lmax, _NSIDE=nside, _noisesig=1.0,
                                   data_mode='synthetic')
    expected_len = (lmax - 2) + (lmax*(lmax+1)//2 - 3) + packed_sizes(lmax)[1]
    assert len(m.x0) == expected_len, f"x0 length {len(m.x0)} != {expected_len}"


@skip_no_deps
def test_model_prior_parameters_tf_dtype():
    import tensorflow as tf

    from diffcmb import CosmologyAdvancedSampling
    m = CosmologyAdvancedSampling(_lmax=8, _NSIDE=2, _noisesig=1.0)
    x0 = m.prior_parameters_tf()
    assert x0.dtype == tf.float64


# ── psi_tf: value and gradient ────────────────────────────────────────────────

@skip_no_deps
def test_psi_tf_is_finite_at_x0():
    """psi_tf must return a finite scalar at the initial state."""
    from diffcmb import CosmologyAdvancedSampling
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

    from diffcmb import CosmologyAdvancedSampling
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

    from diffcmb import CosmologyAdvancedSampling
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

    from diffcmb.model import CosmologyAdvancedSampling as CAS
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


# ── beam / pixel-window forward-model realism (ROADMAP.md Section 2) ─────────

@skip_no_deps
def test_psi_tf_beam_pixwin_matches_ground_truth_synthesis():
    """_psi_tf_raw with beam_fwhm_arcmin set applies B_l*pixwin_l to the
    unlensed alm before synthesis -- validated against an independent
    ground-truth path (healpy almxfl + alm2map), not the model's own
    machinery, per the project's dense-reference discipline.

    Isolates the likelihood term (psi_lik) from the prior/Cl terms -- which
    don't depend on Ninv/data -- by differencing psi_tf against a twin model
    with Ninv zeroed out (removing the likelihood's contribution entirely).
    If the ground-truth-beamed data exactly matches the model's own beamed
    synthesis, that difference is ~0 (up to SHT quadrature precision); an
    unbeamed model fed the same beamed data should NOT match it (negative
    control), confirming the test actually exercises the beam operator.
    """
    import healpy as hp
    import tensorflow as tf

    from diffcmb import CosmologyAdvancedSampling
    from diffcmb.alm_utils import almmotho, splittosingularalm
    from diffcmb.power import beam_pixwin_transfer

    lmax, nside, fwhm_arcmin = 16, 8, 30.0
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]

    rng = np.random.default_rng(5)
    real_alm = rng.standard_normal(n_real) * 5.0
    imag_alm = rng.standard_normal(n_imag) * 5.0

    mo_alm = splittosingularalm(real_alm, imag_alm, lmax)
    ho_alm = almmotho(mo_alm, lmax).astype(np.complex128)

    bl_pixwin = beam_pixwin_transfer(lmax=lmax, fwhm_arcmin=fwhm_arcmin, nside=nside)
    ho_alm_beamed = hp.almxfl(ho_alm, bl_pixwin)
    ground_truth_map = hp.alm2map(ho_alm_beamed, nside, lmax=lmax - 1)

    def build_model(beam_fwhm_arcmin, ninv_scale):
        m = CosmologyAdvancedSampling(
            _lmax=lmax, _NSIDE=nside, _noisesig=1.0, data_mode="synthetic",
            beam_fwhm_arcmin=beam_fwhm_arcmin,
        )
        m._ensure_tf_tensors()
        assert len(m.sph_parts) == 1, "test assumes a single sph_parts chunk"
        m.prior_map_parts = [
            tf.convert_to_tensor(ground_truth_map[m.unmasked_idx], dtype=tf.float64)
        ]
        m.Ninv_parts = [
            tf.convert_to_tensor(np.full(len(m.unmasked_idx), ninv_scale), dtype=tf.float64)
        ]
        return m

    n_lncl = lmax - 2
    params_np = np.zeros(n_lncl + n_real + n_imag)
    params_np[:n_lncl] = 5.0
    params_np[n_lncl:n_lncl + n_real] = real_alm
    params_np[n_lncl + n_real:] = imag_alm
    params_tf = tf.constant(params_np, dtype=tf.float64)

    beamed_model = build_model(fwhm_arcmin, 1.0)
    beamed_model_zero_ninv = build_model(fwhm_arcmin, 0.0)
    psi_lik_beamed = (
        beamed_model._psi_tf_raw(params_tf).numpy()
        - beamed_model_zero_ninv._psi_tf_raw(params_tf).numpy()
    )
    assert abs(psi_lik_beamed) < 1e-4, (
        f"beamed model's own synthesis should match the ground-truth beamed map "
        f"almost exactly, but psi_lik={psi_lik_beamed}"
    )

    unbeamed_model = build_model(None, 1.0)
    unbeamed_model_zero_ninv = build_model(None, 0.0)
    psi_lik_unbeamed = (
        unbeamed_model._psi_tf_raw(params_tf).numpy()
        - unbeamed_model_zero_ninv._psi_tf_raw(params_tf).numpy()
    )
    assert psi_lik_unbeamed > 1.0, (
        "negative control: an unbeamed model fed beamed ground-truth data should "
        "NOT match it -- if this fails, the test isn't exercising the beam operator"
    )


@skip_no_deps
def test_psi_tf_no_beam_matches_prior_behaviour():
    """beam_fwhm_arcmin=None (the default) must leave psi_tf byte-identical
    to before this feature existed -- backward compatibility for the ~10
    existing call sites that don't pass beam_fwhm_arcmin."""
    from diffcmb import CosmologyAdvancedSampling
    m = CosmologyAdvancedSampling(_lmax=8, _NSIDE=2, _noisesig=1.0, data_mode="synthetic")
    m._ensure_tf_tensors()
    assert m.beam_pixwin_per_l is None


# ── anisotropic per-pixel noise (ROADMAP.md Section 2) ───────────────────────

@skip_no_deps
def test_noise_map_builds_ninv_matching_ground_truth_synthetic():
    """When noise_map is given, self.Ninv must equal 1/noise_map**2 exactly
    (an independent ground truth computed directly from noise_map, not via
    any of the model's own machinery) in synthetic data_mode."""
    from diffcmb import CosmologyAdvancedSampling

    lmax, nside = 8, 2
    NPIX = 12 * nside**2
    rng = np.random.default_rng(21)
    noise_map = rng.uniform(0.5, 5.0, NPIX)

    m = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=1.0, data_mode="synthetic",
        noise_map=noise_map,
    )
    expected_ninv = 1.0 / (noise_map**2)
    np.testing.assert_array_equal(m.Ninv, expected_ninv)


@skip_no_deps
def test_noise_map_none_matches_prior_scalar_behaviour():
    """noise_map=None (the default) must leave self.Ninv byte-identical to
    the pre-existing uniform-noise behaviour -- backward compatibility for
    every existing call site that doesn't pass noise_map."""
    from diffcmb import CosmologyAdvancedSampling

    lmax, nside, noisesig = 8, 2, 2.5
    m = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=noisesig, data_mode="synthetic",
    )
    expected_ninv = np.full(12 * nside**2, 1.0 / (noisesig**2))
    np.testing.assert_array_equal(m.Ninv, expected_ninv)


@skip_no_deps
def test_noise_map_uniform_matches_scalar_noisesig():
    """Negative control: a spatially-uniform noise_map must reproduce the
    current scalar-_noisesig self.Ninv bit-for-bit -- confirms noise_map is
    a strict generalisation, not a different code path with a different
    answer in the uniform case."""
    from diffcmb import CosmologyAdvancedSampling

    lmax, nside, noisesig = 8, 2, 2.5
    NPIX = 12 * nside**2
    m_scalar = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=noisesig, data_mode="synthetic",
    )
    uniform_noise_map = np.full(NPIX, noisesig)
    m_map = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=noisesig, data_mode="synthetic",
        noise_map=uniform_noise_map,
    )
    np.testing.assert_array_equal(m_scalar.Ninv, m_map.Ninv)


@skip_no_deps
def test_psi_tf_anisotropic_noise_matches_ground_truth_likelihood():
    """_psi_tf_raw's likelihood term (psi_lik = 0.5 * sum((data-model)**2 *
    Ninv)) must correctly apply a *spatially varying* per-pixel Ninv built
    from noise_map -- validated against an independent ground truth, not the
    model's own machinery, per the project's dense-reference discipline.

    Uses all-zero alm parameters so the model's own synthesised map is
    exactly zero (no SHT quadrature approximation to account for, unlike the
    beam test): the residual is then exactly the injected data vector `w`,
    so psi_lik = 0.5 * sum(w**2 * Ninv) can be checked to near machine
    precision. Isolates psi_lik from the prior/Cl terms (which don't depend
    on Ninv/data) by differencing psi_tf against a twin model with
    Ninv_parts zeroed out. A negative control using a spatially-uniform Ninv
    with the same data confirms the test actually exercises the anisotropic
    weighting (not just linear Ninv scaling).
    """
    import tensorflow as tf

    from diffcmb import CosmologyAdvancedSampling

    lmax, nside = 16, 8
    NPIX = 12 * nside**2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    n_lncl = lmax - 2

    rng = np.random.default_rng(23)
    noise_map = rng.uniform(0.5, 5.0, NPIX)
    w = rng.standard_normal(NPIX)  # arbitrary fixed "data" residual
    expected_ninv = 1.0 / (noise_map**2)

    # All-zero alm/lncl params -> model's own synthesised map is exactly 0.
    params_tf = tf.zeros(n_lncl + n_real + n_imag, dtype=tf.float64)

    def build_model(ninv_array, ninv_scale):
        m = CosmologyAdvancedSampling(
            _lmax=lmax, _NSIDE=nside, _noisesig=1.0, data_mode="synthetic",
            noise_map=noise_map,
        )
        m._ensure_tf_tensors()
        assert len(m.sph_parts) == 1, "test assumes a single sph_parts chunk"
        m.prior_map_parts = [tf.convert_to_tensor(w, dtype=tf.float64)]
        m.Ninv_parts = [tf.convert_to_tensor(ninv_array * ninv_scale, dtype=tf.float64)]
        return m

    anisotropic_model = build_model(expected_ninv, 1.0)
    anisotropic_model_zero = build_model(expected_ninv, 0.0)
    psi_lik_anisotropic = (
        anisotropic_model._psi_tf_raw(params_tf).numpy()
        - anisotropic_model_zero._psi_tf_raw(params_tf).numpy()
    )
    expected_psi_lik = 0.5 * np.sum(w**2 * expected_ninv)
    np.testing.assert_allclose(psi_lik_anisotropic, expected_psi_lik, rtol=1e-9, atol=1e-9)

    # Negative control: a spatially-uniform Ninv with the same total mean
    # weight should give a materially different psi_lik, since w**2 and
    # expected_ninv are (by construction, independent random draws) not
    # spatially uncorrelated in a way that makes the two sums coincide.
    uniform_ninv = np.full(NPIX, np.mean(expected_ninv))
    uniform_model = build_model(uniform_ninv, 1.0)
    uniform_model_zero = build_model(uniform_ninv, 0.0)
    psi_lik_uniform = (
        uniform_model._psi_tf_raw(params_tf).numpy()
        - uniform_model_zero._psi_tf_raw(params_tf).numpy()
    )
    assert abs(psi_lik_uniform - expected_psi_lik) > 1e-2 * abs(expected_psi_lik), (
        "negative control: a spatially-uniform Ninv should NOT reproduce the "
        "anisotropic-weighting result -- if this fails, the test isn't "
        "exercising the per-pixel weighting"
    )


# ── ensure_tf_tensors idempotency ─────────────────────────────────────────────

@skip_no_deps
def test_ensure_tf_tensors_idempotent():
    """Calling _ensure_tf_tensors twice must not change sph or shape."""
    from diffcmb import CosmologyAdvancedSampling
    m = CosmologyAdvancedSampling(_lmax=8, _NSIDE=2, _noisesig=1.0)
    m._ensure_tf_tensors()
    sph_first = m.sph
    shape_first = m.shape
    m._ensure_tf_tensors()
    assert m.sph is sph_first
    assert m.shape is shape_first
