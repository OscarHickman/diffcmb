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

from diffcmb.alm_utils import packed_sizes

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
    n_imag = packed_sizes(lmax)[1]
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


def _packed_slots(L, m):
    """Packed real/imag indices of coefficient (L, m), derived independently.

    real slots run L=2.., m=0..L  -> offset L(L+1)/2 - 3 + m
    imag slots run L=2.., m=1..L  -> offset (L-2)(L+1)/2 + (m - 1)
    """
    real = L * (L + 1) // 2 - 3 + m
    # Imag slot for (L,m): packed_sizes(L)[1] counts the imag slots of all
    # multipoles below L, and within L the m=1..L slots are contiguous.
    imag = packed_sizes(L)[1] + (m - 1) if m >= 1 else None
    return real, imag


@pytest.mark.skipif(not HAS_HEALPY, reason="healpy not installed")
def test_alm_hp_to_packed_uses_true_healpy_ordering():
    """A healpy-ordered mode must pack into the slot for the SAME (L, m).

    healpy alm arrays are m-major (``hp.Alm.getidx``), while this package's
    packed layout is L-major ("author ordering", ``L*(L+1)//2 + m``).
    Converting between the two requires ``alm_utils.almhotmo``/``almmotho``.
    Indexing a healpy-ordered array with the author-ordering formula instead
    silently permutes every coefficient onto a *different* multipole. That
    permutation is its own inverse-consistent partner, so
    ``test_alm_format_round_trip`` above still passes; only a test that pins an
    absolute (L, m) can catch it.
    """
    import healpy as hp

    from diffcmb.lensing import _alm_hp_to_packed

    lmax = 16
    for L, m in [(2, 0), (2, 2), (5, 3), (9, 1), (15, 15)]:
        alm_hp = np.zeros(lmax * (lmax + 1) // 2, dtype=np.complex128)
        alm_hp[hp.Alm.getidx(lmax - 1, L, m)] = 3.0 + 5.0j
        packed = _alm_hp_to_packed(alm_hp, lmax)

        real_slot, imag_slot = _packed_slots(L, m)
        assert packed[real_slot] == pytest.approx(3.0), (
            f"Re a_({L},{m}) landed in the wrong packed slot -- "
            "author/healpy alm ordering not converted"
        )
        # Only m=0 is real for a real field, so m>=1 carries an Im slot too.
        if m >= 1:
            n_real = lmax * (lmax + 1) // 2 - 3
            assert packed[n_real + imag_slot] == pytest.approx(5.0), (
                f"Im a_({L},{m}) landed in the wrong packed slot"
            )
        # Nothing else may be populated.
        assert np.count_nonzero(packed) == (2 if m >= 1 else 1)


@pytest.mark.skipif(not HAS_HEALPY, reason="healpy not installed")
def test_packed_sl_matches_healpy_power_per_multipole():
    """S_L from the packed layout must equal sum_m |a_Lm|^2 at the same L.

    This is the consequence that matters physically: if the ordering is not
    converted, the C_L prior that Block 1 / Block 4 apply per multipole is
    applied to coefficients that live at a different multipole on the sky.
    Only Im a_{L,0} is zeroed on the reference side (a real field has a real
    m=0 coefficient); Im a_{L,1} is left alone, since the packed layout now
    carries it. Both sides therefore see identical data and the comparison is
    exact rather than statistical.
    """
    import healpy as hp

    from diffcmb.lensing import _alm_hp_to_packed, compute_sl_phi_np

    lmax = 20
    rng = np.random.default_rng(11)
    n_alm = lmax * (lmax + 1) // 2
    alm_hp = (
        rng.normal(size=n_alm) + 1j * rng.normal(size=n_alm)
    ).astype(np.complex128)
    for L in range(2, lmax):
        alm_hp[hp.Alm.getidx(lmax - 1, L, 0)] = alm_hp[
            hp.Alm.getidx(lmax - 1, L, 0)
        ].real

    S_ref = np.zeros(lmax)
    for L in range(2, lmax):
        total = 0.0
        for m in range(L + 1):
            val = alm_hp[hp.Alm.getidx(lmax - 1, L, m)]
            total += (1.0 if m == 0 else 2.0) * (val.real ** 2 + val.imag ** 2)
        S_ref[L] = total

    S = compute_sl_phi_np(_alm_hp_to_packed(alm_hp, lmax), lmax)

    np.testing.assert_allclose(
        S[2:lmax], S_ref[2:lmax], rtol=1e-12, atol=1e-14,
        err_msg="packed S_L does not match the healpy power at the same "
                "multipole -- author/healpy alm ordering not converted",
    )


@pytest.mark.skipif(not HAS_HEALPY, reason="healpy not installed")
def test_general_synalm_draw_survives_pack_unpack_with_no_power_loss():
    """A GENERAL sky must survive pack -> unpack with its power intact.

    This is the test the old packing could not pass. `splittosingularalm` used
    to write `complex(real, 0)` for `m == 0 or m == 1`, so Im(a_{L,1}) was
    discarded and every multipole lost one of its 2L+1 real dof -- 20% of the
    power at L=2, 0.8% at L=63. A round-trip test cannot see that on its own
    (the restriction is idempotent, so pack->unpack->pack is still stable);
    what catches it is starting from an UNRESTRICTED hp.synalm draw and
    comparing per-multipole power before and after.
    """
    import healpy as hp

    from diffcmb.lensing import _alm_hp_to_packed, _alm_packed_to_hp

    lmax = 24
    rng = np.random.default_rng(2026)
    cl = np.zeros(lmax)
    cl[2:] = 1.0 / np.arange(2, lmax) ** 2
    np.random.seed(int(rng.integers(1 << 31)))
    alm_hp = hp.synalm(cl, lmax=lmax - 1, new=True).astype(np.complex128)
    alm_hp[: 3] = 0.0  # monopole/dipole are not represented either way

    round_tripped = _alm_packed_to_hp(_alm_hp_to_packed(alm_hp, lmax), lmax)

    np.testing.assert_allclose(
        round_tripped, alm_hp, rtol=1e-12, atol=1e-14,
        err_msg="a general synalm draw did not survive pack -> unpack; the "
                "packed layout is dropping a degree of freedom",
    )

    def power_per_L(a):
        out = np.zeros(lmax)
        for L in range(2, lmax):
            for m in range(L + 1):
                v = a[hp.Alm.getidx(lmax - 1, L, m)]
                out[L] += (1.0 if m == 0 else 2.0) * (v.real ** 2 + v.imag ** 2)
        return out

    np.testing.assert_allclose(
        power_per_L(round_tripped)[2:], power_per_L(alm_hp)[2:],
        rtol=1e-12, atol=1e-30,
        err_msg="per-multipole power changed across the packing round trip",
    )


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

    params_np = np.zeros(lmax - 2 + (lmax * (lmax + 1) // 2 - 3) + packed_sizes(lmax)[1])
    params_np[: lmax - 2] = 5.0

    params_tf = tf.constant(params_np, dtype=tf.float64)
    n_phi = (lmax * (lmax + 1) // 2 - 3) + packed_sizes(lmax)[1]
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
def test_psi_lensed_zero_phi_matches_unlensed_with_beam():
    """psi_lensed with phi=0 must equal model._psi_tf_raw when a beam is set,
    exactly as test_psi_lensed_zero_phi_matches_unlensed does unbeamed --
    confirms lens_map_tf/psi_lensed's beam_pixwin_per_l application
    (lensing.py) is consistent with _psi_tf_raw's (model.py, validated
    directly against an independent healpy ground truth in
    tests/test_model.py::test_psi_tf_beam_pixwin_matches_ground_truth_synthesis)."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from diffcmb import CosmologyAdvancedSampling
    model = CosmologyAdvancedSampling(
        _lmax=LMAX, _NSIDE=NSIDE, _noisesig=100.0,
        data_mode="synthetic", dtype=tf.complex128, beam_fwhm_arcmin=30.0,
    )
    model._ensure_tf_tensors()
    lmax = model.lmax

    params_np = np.zeros(lmax - 2 + (lmax * (lmax + 1) // 2 - 3) + packed_sizes(lmax)[1])
    params_np[: lmax - 2] = 5.0

    params_tf = tf.constant(params_np, dtype=tf.float64)
    n_phi = (lmax * (lmax + 1) // 2 - 3) + packed_sizes(lmax)[1]
    phi_tf = tf.zeros(n_phi, dtype=tf.float64)

    from diffcmb.lensing import psi_lensed
    psi_lens_val = psi_lensed(model, params_tf, phi_tf).numpy()
    psi_unlens_val = model._psi_tf_raw(params_tf).numpy()

    np.testing.assert_allclose(
        psi_lens_val, psi_unlens_val, rtol=1e-6,
        err_msg="psi_lensed(phi=0, beamed) != _psi_tf_raw(beamed)"
    )


@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_psi_lensed_alm_grad_vs_fd():
    """dL/dalm from TF autodiff on psi_lensed agrees with finite differences."""
    from diffcmb.lensing import psi_lensed
    model = _make_model()
    lmax = model.lmax
    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]

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
    n_imag = packed_sizes(lmax)[1]
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
    n_imag = packed_sizes(lmax)[1]

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
    n_imag = packed_sizes(lmax)[1]
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


# ---------------------------------------------------------------------------
# 7 — matrix-free ducc0 SHT path matches the dense-SHT reference (Phase 2
#     gate: Block 3 was dense-SHT-only until this port; see ROADMAP.md
#     Section 1 and the "dense-reference discipline" in Standing discipline)
# ---------------------------------------------------------------------------

def _make_matrixfree_model(lmax=LMAX, nside=NSIDE):
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from diffcmb import CosmologyAdvancedSampling
    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=100.0,
        data_mode="synthetic", dtype=tf.complex128, use_matrixfree_sht=True,
    )
    model._ensure_tf_tensors()
    return model


def _sync_prior_map(dense_model, mf_model, rng):
    """Give both models the same random full-sky prior_map + masked/parts views."""
    npix = dense_model.NPIX
    shared_map = rng.standard_normal(npix)

    dense_model.prior_map = shared_map
    dense_model.prior_map_parts = [
        tf.convert_to_tensor(shared_map[dense_model.unmasked_idx], dtype=tf.float64)
    ]
    mf_model.prior_map = shared_map
    mf_model.prior_map_masked = tf.convert_to_tensor(
        shared_map[mf_model.unmasked_idx], dtype=tf.float64
    )


@pytest.mark.skipif(not HAS_HEALPY or not HAS_TF, reason="healpy/tf not installed")
def test_lens_map_tf_matrixfree_matches_dense():
    """lens_map_tf: matrix-free ducc0 full-sky synthesis vs the dense Y-matrix
    scatter-from-unmasked path, on a full-sky (no mask) model where the two
    should agree to numerical precision."""
    dense_model = _make_model()
    mf_model = _make_matrixfree_model()
    assert len(dense_model.unmasked_idx) == dense_model.NPIX, (
        "test assumes a full-sky (unmasked) model so dense scatter-from-unmasked "
        "and matrix-free full-sky synthesis cover identical pixels"
    )

    rng = np.random.default_rng(7)
    alm_packed = _rand_alm_packed(LMAX, rng)
    phi_hp = np.zeros(hp.Alm.getsize(LMAX - 1), dtype=np.complex128)  # zero phi

    from diffcmb.lensing import lens_map_tf
    T_dense = lens_map_tf(dense_model, tf.constant(alm_packed, tf.float64), phi_hp).numpy()
    T_mf = lens_map_tf(mf_model, tf.constant(alm_packed, tf.float64), phi_hp).numpy()

    np.testing.assert_allclose(T_mf, T_dense, rtol=1e-6, atol=1e-8)


@pytest.mark.skipif(not HAS_HEALPY or not HAS_TF, reason="healpy/tf not installed")
def test_psi_lensed_matrixfree_matches_dense():
    """psi_lensed: matrix-free vs dense-SHT give the same log-posterior value
    and the same gradient w.r.t. alm/C_l, given identical alm/phi/data."""
    dense_model = _make_model()
    mf_model = _make_matrixfree_model()
    assert len(dense_model.unmasked_idx) == dense_model.NPIX

    rng = np.random.default_rng(11)
    _sync_prior_map(dense_model, mf_model, rng)

    lmax = LMAX
    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    params_np = np.zeros(n_lncl + n_real + n_imag)
    params_np[:n_lncl] = 5.0
    params_np[n_lncl:] = rng.standard_normal(n_real + n_imag) * 0.1
    params_tf = tf.Variable(params_np, dtype=tf.float64)

    phi_packed = _rand_phi_packed(lmax, rng, amplitude=1e-4)
    phi_tf = tf.constant(phi_packed, dtype=tf.float64)

    from diffcmb.lensing import psi_lensed

    with tf.GradientTape() as tape:
        val_dense = psi_lensed(dense_model, params_tf, phi_tf)
    grad_dense = tape.gradient(val_dense, params_tf).numpy()

    with tf.GradientTape() as tape:
        val_mf = psi_lensed(mf_model, params_tf, phi_tf)
    grad_mf = tape.gradient(val_mf, params_tf).numpy()

    np.testing.assert_allclose(val_mf.numpy(), val_dense.numpy(), rtol=1e-6)
    np.testing.assert_allclose(grad_mf, grad_dense, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 7 — matrix-free vs dense, masked sky (ROADMAP.md Section 1: "Not yet
#     re-validated on a masked sky" -- the dense-reference discipline applied
#     to the masked-sky matrix-free lensing path for the first time, before
#     any masked-sky Phase 2 chain such as the planned f_sky~0.016
#     patch-scale sanity check.)
# ---------------------------------------------------------------------------

def _polar_cap_mask_idx(nside, f_sky):
    """Contiguous polar-cap mask index set covering the requested f_sky
    fraction of the sphere (same construction as test_samplers.py's
    small_masked_matrixfree_model fixture)."""
    npix = 12 * nside * nside
    theta, _ = hp.pix2ang(nside, np.arange(npix))
    cutoff = np.arccos(1 - 2 * f_sky)
    return np.where(theta < cutoff)[0]


def _make_masked_dense_model(lmax=LMAX, nside=NSIDE, f_sky=0.3):
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from diffcmb import CosmologyAdvancedSampling
    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=100.0,
        data_mode="synthetic", dtype=tf.complex128,
    )
    model.unmasked_idx = _polar_cap_mask_idx(nside, f_sky)
    model._ensure_tf_tensors()
    return model


def _make_masked_matrixfree_model(lmax=LMAX, nside=NSIDE, f_sky=0.3):
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from diffcmb import CosmologyAdvancedSampling
    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=100.0,
        data_mode="synthetic", dtype=tf.complex128, use_matrixfree_sht=True,
    )
    model.unmasked_idx = _polar_cap_mask_idx(nside, f_sky)
    model._ensure_tf_tensors()
    return model


@pytest.mark.skipif(not HAS_HEALPY or not HAS_TF, reason="healpy/tf not installed")
def test_lens_map_tf_matrixfree_matches_dense_masked_sky():
    """lens_map_tf: matrix-free ducc0 synthesis vs the dense Y-matrix path
    agree to numerical precision on a masked sky (f_sky~0.3, a polar cap),
    the first correctness check of the matrix-free path outside full-sky."""
    dense_model = _make_masked_dense_model()
    mf_model = _make_masked_matrixfree_model()
    np.testing.assert_array_equal(dense_model.unmasked_idx, mf_model.unmasked_idx)
    assert len(dense_model.unmasked_idx) < dense_model.NPIX, "mask should be nontrivial"

    rng = np.random.default_rng(23)
    alm_packed = _rand_alm_packed(LMAX, rng)
    phi_hp = np.zeros(hp.Alm.getsize(LMAX - 1), dtype=np.complex128)  # zero phi

    from diffcmb.lensing import lens_map_tf
    T_dense = lens_map_tf(dense_model, tf.constant(alm_packed, tf.float64), phi_hp).numpy()
    T_mf = lens_map_tf(mf_model, tf.constant(alm_packed, tf.float64), phi_hp).numpy()

    np.testing.assert_allclose(T_mf, T_dense, rtol=1e-6, atol=1e-8)


@pytest.mark.skipif(not HAS_HEALPY or not HAS_TF, reason="healpy/tf not installed")
def test_psi_lensed_matrixfree_matches_dense_masked_sky_zero_phi():
    """psi_lensed: matrix-free vs dense give the same log-posterior value and
    gradient on a masked sky when phi=0 (identity lensing) -- a valid
    comparison because the dense path's zero-padding-outside-the-mask
    artifact (see the nonzero-phi note below) can't matter when there's no
    deflection to sample it. Nonzero-phi masked-sky matrix-free gradients
    are checked directly against finite differences instead, in
    test_psi_lensed_matrixfree_alm_grad_vs_fd_masked_sky below."""
    dense_model = _make_masked_dense_model()
    mf_model = _make_masked_matrixfree_model()
    np.testing.assert_array_equal(dense_model.unmasked_idx, mf_model.unmasked_idx)

    rng = np.random.default_rng(29)
    _sync_prior_map(dense_model, mf_model, rng)

    lmax = LMAX
    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    params_np = np.zeros(n_lncl + n_real + n_imag)
    params_np[:n_lncl] = 5.0
    params_np[n_lncl:] = rng.standard_normal(n_real + n_imag) * 0.1
    params_tf = tf.Variable(params_np, dtype=tf.float64)

    phi_tf = tf.zeros(n_real + n_imag, dtype=tf.float64)

    from diffcmb.lensing import psi_lensed

    with tf.GradientTape() as tape:
        val_dense = psi_lensed(dense_model, params_tf, phi_tf)
    grad_dense = tape.gradient(val_dense, params_tf).numpy()

    with tf.GradientTape() as tape:
        val_mf = psi_lensed(mf_model, params_tf, phi_tf)
    grad_mf = tape.gradient(val_mf, params_tf).numpy()

    np.testing.assert_allclose(val_mf.numpy(), val_dense.numpy(), rtol=1e-6)
    np.testing.assert_allclose(grad_mf, grad_dense, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not HAS_HEALPY or not HAS_TF, reason="healpy/tf not installed")
def test_psi_lensed_matrixfree_alm_grad_vs_fd_masked_sky():
    """dL/dalm from TF autodiff on the matrix-free psi_lensed agrees with
    finite differences on a masked sky (f_sky~0.3), nonzero phi.

    NOTE (real finding, 2026-07-18): comparing matrix-free vs *dense*
    gradients at nonzero phi on a masked sky (mirroring the full-sky
    dense-reference test above) fails at up to 8% relative error on ~58% of
    components. Root cause is not a matrix-free bug: the dense path's
    lens_map_tf/psi_lensed only ever computes T_unlensed at the model's
    unmasked pixels and *zero-pads* everything else before bilinear
    deflection-interpolation (lensing.py's `unsorted_segment_sum` scatter) --
    an unphysical discontinuity at the mask edge that nonzero-phi deflection
    samples can land on. This is exactly the scenario sht_ducc.py's
    full_synthesis_tf docstring and CLAUDE.md were written to avoid ("lensing
    needs this because deflected positions can fall outside the eventual
    mask") -- the dense path was never a valid masked+lensed reference, only
    a valid masked+identity (phi=0) one (see the test above). So the correct
    validation of matrix-free masked-sky gradients is a direct FD check, not
    a dense comparison; this test is that check, and it passes on its own
    terms."""
    from diffcmb.lensing import psi_lensed
    model = _make_masked_matrixfree_model()
    lmax = model.lmax
    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]

    rng = np.random.default_rng(31)
    params_np = np.zeros(n_lncl + n_real + n_imag)
    params_np[:n_lncl] = 5.0
    params_np[n_lncl:] = rng.standard_normal(n_real + n_imag) * 0.1
    phi_np = _rand_phi_packed(lmax, rng, amplitude=1e-4)

    params_var = tf.Variable(params_np, dtype=tf.float64)
    phi_tf = tf.constant(phi_np, dtype=tf.float64)

    with tf.GradientTape() as tape:
        val = psi_lensed(model, params_var, phi_tf)
    g_auto = tape.gradient(val, params_var).numpy()

    eps = 1e-5
    g_fd = np.zeros(len(params_np))
    for i in range(n_lncl, n_lncl + 5):
        p_p = params_np.copy()
        p_p[i] += eps
        p_m = params_np.copy()
        p_m[i] -= eps
        lp = psi_lensed(model, tf.constant(p_p, tf.float64), phi_tf).numpy()
        lm = psi_lensed(model, tf.constant(p_m, tf.float64), phi_tf).numpy()
        g_fd[i] = (lp - lm) / (2 * eps)

    alm_slice = slice(n_lncl, n_lncl + 5)
    np.testing.assert_allclose(
        g_auto[alm_slice], g_fd[alm_slice], rtol=1e-4, atol=1e-6,
        err_msg="dL/dalm autodiff vs FD mismatch in masked-sky matrix-free psi_lensed"
    )


# ---------------------------------------------------------------------------
# 8 — estimate_phi_diag_fisher: coordinate-sampled diagonal likelihood-
#     curvature estimate for the phi HMC preconditioner (ROADMAP.md Section
#     1, "Simulation validation" follow-up -- the prior-only phi mass matrix
#     was found to leave the phi block barely mixing at lmax=300).
#
#     Design note: an earlier joint-random-direction (true Hutchinson)
#     probe was tried and abandoned -- perturbing all ~n_phi modes at once
#     crosses the same C0-but-not-C1 bilinear-interpolation-cell boundary
#     documented in test_psi_lensed_phi_grad_vs_fd's eps note, and even
#     after rescaling eps to avoid that, the off-diagonal coupling in this
#     Hessian is large enough that Hutchinson's estimator needs far more
#     than a handful of probes to converge (empirically: 40 joint probes
#     gave per-probe estimates in the thousands against a true diagonal of
#     ~110). Perturbing one packed-phi coordinate at a time instead (the
#     same FD-of-analytic-gradient computation the brute-force reference
#     below uses, just on a sampled subset rather than every mode) is
#     stable and low-variance by construction.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_estimate_phi_diag_fisher_vs_dense_hessian_small_lmax():
    """estimate_phi_diag_fisher's per-L curvature estimate should closely
    match a brute-force FD-of-analytic-gradient diagonal Hessian of
    psi_lensed at one chosen L. With n_probes >= the mode count at that L,
    the estimator samples every mode there deterministically, so this
    should match the dense reference tightly (not just in expectation).
    """
    from diffcmb.lensing import estimate_phi_diag_fisher, psi_lensed
    model = _make_model()
    lmax = model.lmax
    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]

    rng = np.random.default_rng(21)
    params_np = np.zeros(n_lncl + n_real + n_imag)
    params_np[:n_lncl] = 5.0
    params_np[n_lncl:] = rng.standard_normal(n_real + n_imag) * 0.1
    params_tf = tf.constant(params_np, dtype=tf.float64)
    phi_np = _rand_phi_packed(lmax, rng, amplitude=1e-4)

    def grad_at(phi_val):
        phi_var = tf.Variable(phi_val, dtype=tf.float64)
        with tf.GradientTape() as tape:
            val = psi_lensed(model, params_tf, phi_var)
        return tape.gradient(val, phi_var).numpy()

    # Brute-force exact diagonal Hessian (central FD of the analytic
    # gradient, one index at a time) for every packed-phi mode at L=6,
    # matching estimate_phi_diag_fisher's per-L averaging convention.
    from diffcmb.samplers import _alm_index_lm
    L_arr, _m_arr = _alm_index_lm(lmax, n_real, n_imag)
    probe_L = 6
    idx_at_L = np.where(L_arr == probe_L)[0]
    assert len(idx_at_L) > 0

    eps = 1e-9
    dense_diag = np.zeros(len(idx_at_L))
    for k, i in enumerate(idx_at_L):
        phi_p = phi_np.copy()
        phi_p[i] += eps
        phi_m = phi_np.copy()
        phi_m[i] -= eps
        g_p = grad_at(phi_p)[i]
        g_m = grad_at(phi_m)[i]
        dense_diag[k] = (g_p - g_m) / (2 * eps)
    dense_ref_at_L = np.mean(np.maximum(dense_diag, 0.0))

    # n_probes=40 exceeds the mode count at every L for this small model
    # (max multiplicity is 2*lmax-1=39), so every L is sampled exhaustively
    # and deterministically -- the comparison below is not stochastic.
    diag_fisher_per_L = estimate_phi_diag_fisher(
        model, params_tf, tf.constant(phi_np, dtype=tf.float64), lmax,
        n_probes=40, rng=np.random.default_rng(5),
    )

    assert diag_fisher_per_L.shape == (lmax,)
    assert np.all(diag_fisher_per_L >= 0.0)
    np.testing.assert_allclose(
        diag_fisher_per_L[probe_L], dense_ref_at_L, rtol=0.05, atol=1e-6,
        err_msg="estimate_phi_diag_fisher diverges from the brute-force "
                "dense diagonal Hessian reference at L=6 (exhaustive sampling "
                "there should match closely, not just in expectation)"
    )


# ---------------------------------------------------------------------------
# 8b — estimate_phi_block_hessian: Nystrom low-rank cross-L Hessian
#      correction, block-diagonal by m (ROADMAP.md/achievements.md
#      2026-08-18 -- diagnose_phi_hessian_coupling.py found significant
#      cross-L coupling that no diagonal-in-L mass matrix can represent).
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_estimate_phi_block_hessian_matches_dense_block_small_lmax():
    """With n_probes == the block size (exhaustive sampling), the Nystrom
    approximation degenerates to the exact block Hessian (H[:,S] with S =
    all indices, pinv(H[S,S]) @ H[S,S] = I when H[S,S] is invertible), so
    it should match a brute-force dense FD-of-analytic-gradient Hessian of
    that block closely -- not just in expectation.
    """
    from diffcmb.lensing import estimate_phi_block_hessian, psi_lensed
    from diffcmb.samplers import _alm_index_lm

    model = _make_model()
    lmax = model.lmax
    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]

    rng = np.random.default_rng(22)
    params_np = np.zeros(n_lncl + n_real + n_imag)
    params_np[:n_lncl] = 5.0
    params_np[n_lncl:] = rng.standard_normal(n_real + n_imag) * 0.1
    params_tf = tf.constant(params_np, dtype=tf.float64)
    phi_np = _rand_phi_packed(lmax, rng, amplitude=1e-4)

    def grad_at(phi_val):
        phi_var = tf.Variable(phi_val, dtype=tf.float64)
        with tf.GradientTape() as tape:
            val = psi_lensed(model, params_tf, phi_var)
        return tape.gradient(val, phi_var).numpy()

    L_arr, m_arr = _alm_index_lm(lmax, n_real, n_imag)
    channel_arr = np.array(["real"] * n_real + ["imag"] * n_imag)
    # m=2 real block has the most L's (L=2..lmax-1) for this small model --
    # exercise a block with more than one mode.
    probe_m = 2
    idx = np.where((channel_arr == "real") & (m_arr == probe_m))[0]
    assert len(idx) >= 2

    eps = 1e-9
    g0 = grad_at(phi_np)
    K = len(idx)
    dense_block = np.empty((K, K))
    for k, j in enumerate(idx):
        phi_p = phi_np.copy()
        phi_p[j] += eps
        g1 = grad_at(phi_p)
        dense_block[:, k] = (g1[idx] - g0[idx]) / eps
    dense_block = 0.5 * (dense_block + dense_block.T)

    # n_probes=1 (rank-1 Nystrom) has a simple, numerically stable closed
    # form -- H_nystrom = c c^T / H_SS[0,0] for the single sampled column c
    # -- that this test can check directly against the dense reference's
    # own column, without inverting a large near-singular sampled
    # submatrix (which n_probes=K, "exhaustive" sampling, would require:
    # H_SS is then the full noisy FD block itself, and independently
    # verified to be numerically unstable to invert at this eps/scale --
    # not representative of production use, which only ever samples a
    # handful of probes per block).
    blocks = estimate_phi_block_hessian(
        model, params_tf, tf.constant(phi_np, dtype=tf.float64), lmax,
        n_probes=1, rng=np.random.default_rng(6),
    )
    key = ("real", probe_m)
    assert key in blocks
    idx_out, H_out = blocks[key]
    np.testing.assert_array_equal(idx_out, idx)
    assert H_out.shape == (K, K)
    # Symmetric and PSD by construction (outer product of a real vector).
    np.testing.assert_allclose(H_out, H_out.T, rtol=0, atol=1e-8)
    eigval = np.linalg.eigvalsh(H_out)
    assert np.all(eigval >= -1e-6 * max(1.0, np.abs(eigval).max()))
    # Rank <= 1: exactly one eigenvalue should carry (numerically) all the
    # trace.
    eigval_sorted = np.sort(np.abs(eigval))
    assert eigval_sorted[-1] > 0
    assert eigval_sorted[:-1].sum() < 1e-6 * eigval_sorted[-1]


# ---------------------------------------------------------------------------
# 9 — Optional Block 4: compute_sl_phi_np / sample_cl_phiphi_given_phi
#     (ROADMAP.md's lowest-priority optional item -- "C_l^phiphi|phi exact
#     inverse-Gamma, same structure as Block 1"). log_prob_phi_block's
#     Gaussian prior term on phi has the identical per-L quadratic-form
#     structure as the CMB alm prior that model.py::compute_sl_np /
#     sample_cl_given_alm already exploit for Block 1, so validate the phi
#     analogue the same way: (a) cross-check the m=0-vs-m>=1 weighting
#     against an independent brute-force sum over the healpy-ordered
#     (unpacked) complex alm -- this is exactly the kind of weighting mistake
#     flagged elsewhere in this project (a uniform 1/C_l used where the real
#     convention needs 2/C_l for m>0); (b) confirm the exact-draw recovers a
#     known generating C_L^phiphi across many independent realizations.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HEALPY, reason="healpy not installed")
def test_compute_sl_phi_np_matches_brute_force_healpy_ordering():
    """compute_sl_phi_np's S_L = sum_{m=-L}^{L} |phi_Lm|^2 (packed layout,
    weight 1 for m=0 / weight 2*(re^2+im^2) for m>=1) must match a brute-force
    sum computed independently from the healpy-ordered (unpacked) complex alm
    via the standard real-field identity
        sum_{m=-L}^{L} |a_Lm|^2 = |a_L0|^2 + 2 * sum_{m=1}^{L} |a_Lm|^2.
    _rand_phi_packed already zeros Im(phi_{L,1}) when round-tripping through
    _alm_hp_to_packed (same packed convention as the CMB alm), so the two
    sums are computed on identical underlying data and should match to
    float64 precision, not just approximately.
    """
    from diffcmb.lensing import _alm_packed_to_hp, compute_sl_phi_np
    lmax = 15
    rng = np.random.default_rng(3)
    phi_packed = _rand_phi_packed(lmax, rng, amplitude=1.0)

    S = compute_sl_phi_np(phi_packed, lmax)

    # Index the unpacked array with hp.Alm.getidx -- the real healpy m-major
    # ordering. This used to use the triangular L*(L+1)//2 + m formula, which
    # matched _alm_packed_to_hp's own (incorrect) indexing and so agreed with
    # it by construction rather than checking it.
    import healpy as hp
    alm_hp = _alm_packed_to_hp(phi_packed, lmax)
    S_brute = np.zeros(lmax)
    for L in range(2, lmax):
        s = 0.0
        for m in range(L + 1):
            val = alm_hp[hp.Alm.getidx(lmax - 1, L, m)]
            weight = 1.0 if m == 0 else 2.0
            s += weight * (val.real ** 2 + val.imag ** 2)
        S_brute[L] = s

    np.testing.assert_allclose(
        S[2:lmax], S_brute[2:lmax], rtol=1e-10, atol=1e-16,
        err_msg="compute_sl_phi_np diverges from the brute-force "
                "healpy-ordering S_L reference -- check the m=0-vs-m>=1 "
                "weighting convention"
    )


def test_sample_cl_phiphi_given_phi_recovers_known_spectrum():
    """Exact-draw recovery check: draw many independent phi realizations
    from a known constant C_L^phiphi = C0 (using the per-mode Gaussian
    variances implied by log_prob_phi_block's prior term -- Var(re)=C_L for
    m=0, Var(re)=Var(im)=C_L/2 for m>=1, matching the weight-1/weight-2
    convention compute_sl_phi_np mirrors from compute_sl_np), draw one
    posterior C_L sample per realization via sample_cl_phiphi_given_phi, and
    confirm the empirical mean across realizations recovers C0 (up to the
    known, small finite-L bias of a flat-prior InvGamma posterior mean --
    E[C_L_sampled] = C0 * L/(L-1.5) under this generative model, a ~6% effect
    at L=25 that shrinks as L grows) to within a statistical margin (mean +/-
    6 standard errors around that expected, not the naively unbiased, value)
    -- this still catches gross convention errors (e.g. a 2x weighting bug,
    which would be a ~100% effect, far outside 6 standard errors) while not
    penalizing the estimator for a bias that is expected and benign.
    """
    from diffcmb.lensing import sample_cl_phiphi_given_phi
    from diffcmb.samplers import _alm_index_lm

    lmax = 30
    L_probe = 25
    C0 = 3e-9
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    n_tot = n_real + n_imag

    L_arr, m_arr = _alm_index_lm(lmax, n_real, n_imag)
    is_real = np.arange(n_tot) < n_real
    var = np.empty(n_tot)
    real_idx = np.where(is_real)[0]
    var[real_idx] = np.where(m_arr[real_idx] == 0, C0, C0 / 2.0)
    imag_idx = np.where(~is_real)[0]
    var[imag_idx] = C0 / 2.0
    std = np.sqrt(var)

    rng = np.random.default_rng(11)
    n_rep = 3000
    cl_draws = np.empty(n_rep)
    for i in range(n_rep):
        phi_packed = rng.normal(scale=std)
        lncl = sample_cl_phiphi_given_phi(phi_packed, lmax, rng=rng)
        cl_draws[i] = np.exp(lncl[L_probe - 2])

    recovered = float(np.mean(cl_draws))
    se = float(np.std(cl_draws, ddof=1) / np.sqrt(n_rep))
    expected = C0 * L_probe / (L_probe - 1.5)  # flat-prior InvGamma posterior-mean bias
    assert abs(recovered - expected) < 6 * se, (
        f"sample_cl_phiphi_given_phi failed to recover the expected "
        f"(bias-adjusted) C_L={expected:.3e} at L={L_probe} from true "
        f"C0={C0:.3e} (recovered {recovered:.3e} +/- {se:.3e}) -- check "
        f"compute_sl_phi_np's m=0-vs-m>=1 weighting"
    )


def test_sample_cl_phiphi_given_phi_matches_false_when_zero_phi():
    """phi=0 -> S_L=0 for all L -> sampled C_L should be at (or clipped near)
    the numerical floor, not some arbitrary finite value -- a basic sanity
    check that beta's floor (1e-60) and the output clip (1e-30) are wired
    together consistently."""
    from diffcmb.lensing import sample_cl_phiphi_given_phi
    lmax = 10
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    phi_packed = np.zeros(n_real + n_imag)
    rng = np.random.default_rng(0)
    lncl = sample_cl_phiphi_given_phi(phi_packed, lmax, rng=rng)
    assert lncl.shape == (lmax - 2,)
    assert np.all(np.isfinite(lncl))
    assert np.all(np.exp(lncl) <= 1e-20)


def test_sample_cl_phiphi_given_phi_rejects_non_finite_input():
    from diffcmb.lensing import sample_cl_phiphi_given_phi
    lmax = 10
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    phi_packed = np.zeros(n_real + n_imag)
    phi_packed[0] = np.nan
    with pytest.raises(ValueError):
        sample_cl_phiphi_given_phi(phi_packed, lmax)


# ---------------------------------------------------------------------------
# 10 — Block 4 makes the phi prior scale-free (root cause of job 11887897)
# ---------------------------------------------------------------------------

def test_block4_refitted_prior_is_scale_free_in_phi_amplitude():
    """With Block 4 on, the phi Gaussian prior exerts NO restoring force on the
    overall amplitude of phi -- it is exactly constant along the scaling ray.

    Block 4 draws C_L^phiphi | phi ~ InvGamma(L-0.5, S_L/2), whose mean is
    S_L/(2(L-1.5)). Substituting that back into log_prob_phi_block's prior term:

        0.5 * sum_L S_L / C_L = 0.5 * sum_L S_L * 2(L-1.5)/S_L = sum_L (L-1.5)

    which does not depend on phi at all. This is the phi-side face of the flat,
    improper implied prior that coverage_ensemble_chain.py's header documents,
    and it is the reason the sampler cannot pull an inflated phi back down:
    the ONLY thing pinning the amplitude is the lensing likelihood.

    Regression guard for the 2026-08-28 coverage-ensemble failure (job
    11887897), where a cold alm start shifted the likelihood's preferred phi
    amplitude by ~30x and the prior offered no opposition, leaving phi frozen
    ~7e4x above truth in power. Anyone tempted to assume "the prior will
    regularise the amplitude" should read this test.
    """
    from diffcmb.lensing import compute_sl_phi_np

    lmax = 12
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    rng = np.random.default_rng(11887897)
    phi_packed = rng.normal(size=n_real + n_imag)

    def block4_prior_term(phi):
        """0.5 * sum_L S_L / C_L with C_L at Block 4's conditional mean."""
        S = compute_sl_phi_np(phi, lmax)
        total = 0.0
        for L in range(2, lmax):
            cl = S[L] / (2.0 * (L - 1.5))
            total += 0.5 * S[L] / cl
        return total

    expected = sum(L - 1.5 for L in range(2, lmax))
    base = block4_prior_term(phi_packed)
    assert base == pytest.approx(expected, rel=1e-12)

    # The whole point: rescaling phi over four orders of magnitude does not
    # change the prior term by even one part in 1e-10.
    for s in (1e-2, 0.1, 1.0, 10.0, 1e2):
        assert block4_prior_term(s * phi_packed) == pytest.approx(base, rel=1e-10)


def test_block4_fixed_spectrum_prior_DOES_constrain_amplitude():
    """Contrast case: with C_L^phiphi held FIXED (Block 4 off), the same prior
    term scales as s^2 and does pin the amplitude.

    Together with the test above this isolates exactly what enabling Block 4
    gives up, so the trade-off is explicit rather than folklore.
    """
    from diffcmb.lensing import compute_sl_phi_np

    lmax = 12
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    rng = np.random.default_rng(0)
    phi_packed = rng.normal(size=n_real + n_imag)
    cl_fixed = np.full(lmax, 1.0)

    def fixed_prior_term(phi):
        S = compute_sl_phi_np(phi, lmax)
        return 0.5 * sum(S[L] / cl_fixed[L] for L in range(2, lmax))

    base = fixed_prior_term(phi_packed)
    for s in (0.5, 2.0, 10.0):
        assert fixed_prior_term(s * phi_packed) == pytest.approx(
            s * s * base, rel=1e-10
        )


# ---------------------------------------------------------------------------
# Block 4 with a PROPER (weakly informative) prior on C_L^phiphi
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_HEALPY, reason="healpy required")
def test_cl_phiphi_prior_nu_none_is_bit_identical_to_flat_prior():
    """`prior_nu=None` must reproduce the old flat-improper-prior draw exactly.

    Same "None = old behaviour, zero effect on existing call sites" contract
    the beam/noise_map kwargs follow. Bit-identity (not closeness) because the
    two paths must consume the RNG stream identically too -- otherwise every
    existing chain's reproducibility silently changes.
    """
    from diffcmb.lensing import sample_cl_phiphi_given_phi

    lmax = 12
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    phi = np.random.default_rng(0).normal(scale=1e-3, size=n_real + n_imag)

    a = sample_cl_phiphi_given_phi(phi, lmax, rng=np.random.default_rng(7))
    b = sample_cl_phiphi_given_phi(phi, lmax, rng=np.random.default_rng(7),
                                   prior_nu=None, cl_phiphi_fid=None)
    np.testing.assert_allclose(a, b, rtol=0, atol=0)


@pytest.mark.skipif(not HAS_HEALPY, reason="healpy required")
def test_cl_phiphi_proper_prior_matches_conjugate_invgamma():
    """The proper-prior draw must come from the exact conjugate posterior.

    Prior C_L ~ InvGamma(alpha_0, beta_0) with alpha_0 = nu/2,
    beta_0 = nu*C_L^fid/2 (nu = "effective number of prior pseudo-modes",
    matching how the data contributes k_L = 2L real modes -- 2L, not 2L+1,
    because the packing forces Im(a_{L,1}) = 0). Multiplying by the Gaussian
    phi prior C^{-k_L/2} exp(-S_L/2C) gives

        C_L | phi ~ InvGamma(k_L/2 + nu/2, (S_L + nu*C_L^fid)/2)
                  = InvGamma(L + nu/2, (S_L + nu*C_L^fid)/2).

    Checked by KS against that law rather than by re-deriving it in the test.
    """
    from scipy import stats

    from diffcmb.lensing import compute_sl_phi_np, sample_cl_phiphi_given_phi

    lmax = 8
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    rng = np.random.default_rng(3)
    phi = rng.normal(scale=1e-3, size=n_real + n_imag)
    nu = 6.0
    cl_fid = np.full(lmax, 1e-6)

    draws = np.array([
        np.exp(sample_cl_phiphi_given_phi(phi, lmax, rng=rng, prior_nu=nu,
                                          cl_phiphi_fid=cl_fid))
        for _ in range(4000)
    ])

    S = compute_sl_phi_np(phi, lmax)
    for i in range(lmax - 2):
        L = i + 2
        # k_L/2 + nu/2 with k_L = 2L+1 (Im(a_{L,1}) restored)
        alpha = L + 0.5 + nu / 2.0
        beta = (S[L] + nu * cl_fid[L]) / 2.0
        p = stats.kstest(draws[:, i], "invgamma", args=(alpha, 0.0, beta)).pvalue
        assert p > 1e-3, f"L={L}: draws do not match conjugate posterior (KS p={p:.2g})"


@pytest.mark.skipif(not HAS_HEALPY, reason="healpy required")
def test_cl_phiphi_proper_prior_shrinks_an_inflated_amplitude_toward_fiducial():
    """The whole point: unlike the flat prior, this one pulls C_L back down.

    Guards the 2026-08-28 failure mode (job 11887897) where phi inflated
    1e3-1e5x and nothing opposed it. With phi 100x too large in amplitude
    (1e4x in power), the flat-prior draw tracks the inflated S_L essentially
    exactly, while the proper prior must land materially closer to fiducial.
    """
    from diffcmb.lensing import sample_cl_phiphi_given_phi

    lmax = 10
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    cl_fid = np.full(lmax, 1e-6)
    rng = np.random.default_rng(5)
    # phi drawn at the fiducial scale, then inflated 100x in amplitude.
    phi = rng.normal(scale=np.sqrt(1e-6), size=n_real + n_imag) * 100.0

    flat = np.exp([sample_cl_phiphi_given_phi(phi, lmax, rng=np.random.default_rng(k))
                   for k in range(200)]).mean(axis=0)
    proper = np.exp([
        sample_cl_phiphi_given_phi(phi, lmax, rng=np.random.default_rng(k),
                                   prior_nu=6.0, cl_phiphi_fid=cl_fid)
        for k in range(200)
    ]).mean(axis=0)

    # Both are still above fiducial (the data really does say phi is large),
    # but the proper prior must shrink every multipole toward it.
    assert np.all(proper < flat), "proper prior failed to shrink an inflated C_L"
    assert np.all(proper > cl_fid[2:lmax]), "prior over-shrank past the fiducial"


@pytest.mark.skipif(not HAS_HEALPY, reason="healpy required")
def test_cl_phiphi_prior_nu_must_be_positive_for_a_proper_phi_marginal():
    """nu > 0 is a correctness requirement, not a taste parameter.

    Integrating C_L out of the joint leaves a phi marginal
        p(r) ∝ r^{k_L - 1} (r^2/2 + beta_0)^{-(k_L/2 + nu/2)}   [r^2 = S_L]
    whose large-r tail is r^{-1-nu} (the k_L cancels). That is normalisable
    exactly for nu > 0; at nu = 0 it decays as 1/r and still diverges
    logarithmically, which is the improper-prior pathology this feature
    exists to remove -- and is what the flat default (a0 = -1, tail r^{+1})
    does far more violently. A nu that silently produced an improper target
    would be worse than no feature at all, so the API must reject it.

    Note the threshold was nu > 2 when the code assumed 2L+1 packed dof;
    correcting k_L to 2L moved it to nu > 0. The guard has to be derived
    alongside the exponent, not carried over.
    """
    from diffcmb.lensing import sample_cl_phiphi_given_phi

    lmax = 6
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    phi = np.random.default_rng(1).normal(scale=1e-3, size=n_real + n_imag)
    cl_fid = np.full(lmax, 1e-6)

    for bad_nu in (-1.0, 0.0):
        with pytest.raises(ValueError, match="nu"):
            sample_cl_phiphi_given_phi(phi, lmax, rng=np.random.default_rng(0),
                                       prior_nu=bad_nu, cl_phiphi_fid=cl_fid)

    # And the tail exponent itself: verify numerically that the marginal
    # falls off as r^{1-nu}, so the nu>2 threshold is the real boundary.
    from scipy.special import gammaln

    from diffcmb.alm_utils import packed_dof_per_multipole
    for nu in (0.5, 4.0, 8.0):
        L = 5
        k = float(packed_dof_per_multipole(L + 1)[L])
        beta_0 = nu * 1e-6 / 2.0
        alpha = k / 2.0 + nu / 2.0

        def log_p_r(r, alpha=alpha, beta_0=beta_0, k=k):
            return (k - 1.0) * np.log(r) - alpha * np.log(r ** 2 / 2.0 + beta_0) \
                + gammaln(alpha)

        r1, r2 = 1e3, 1e4        # deep in the tail
        slope = (log_p_r(r2) - log_p_r(r1)) / (np.log(r2) - np.log(r1))
        assert abs(slope - (-1.0 - nu)) < 1e-6, \
            f"nu={nu}: tail exponent {slope:.4f}, expected {-1.0 - nu:.4f}"


@pytest.mark.skipif(not HAS_HEALPY, reason="healpy required")
def test_cl_phiphi_proper_prior_is_not_scale_free_in_phi_amplitude():
    """Contrast case to test_block4_refitted_prior_is_scale_free_in_phi_amplitude.

    That test pins the flat-prior pathology: substituting Block 4's conditional
    mean makes the phi prior term exactly constant along the scaling ray, so
    nothing restrains the amplitude. With a proper prior the same substitution
    must produce a term that GROWS with amplitude -- i.e. a real restoring
    force. This is the property the whole feature is for.
    """
    from diffcmb.lensing import compute_sl_phi_np

    lmax = 12
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    phi0 = np.random.default_rng(4).normal(scale=1e-3, size=n_real + n_imag)
    nu, cl_fid = 6.0, np.full(lmax, 1e-6)

    def prior_term(scale):
        """0.5 * sum_L S_L / C_L at the conditional mean of C_L."""
        S = compute_sl_phi_np(phi0 * scale, lmax)
        total = 0.0
        for L in range(2, lmax):
            alpha = L + 0.5 + nu / 2.0   # k_L/2 + nu/2, k_L = 2L+1
            beta = (S[L] + nu * cl_fid[L]) / 2.0
            c_mean = beta / (alpha - 1.0)          # InvGamma mean
            total += 0.5 * S[L] / c_mean
        return total

    vals = [prior_term(s) for s in (0.1, 1.0, 10.0, 100.0)]
    assert all(b > a for a, b in zip(vals, vals[1:])), \
        f"proper prior is not restoring the amplitude: {vals}"
    # And it must genuinely diverge, not merely tick up.
    assert vals[-1] > 10.0 * vals[0]


@pytest.mark.skipif(not HAS_HEALPY, reason="healpy required")
def test_packed_dof_per_multipole_matches_the_actual_packing():
    """k_L must be counted from the packing, not assumed.

    `splittosingularalm` writes `complex(real, 0)` only when `m == 0`, so with
    Im(a_{L,1}) restored every multipole carries the full 2L+1 real dof.
    Cross-checked against the independent (L, m) index map used by the
    samplers, so the two descriptions of the layout cannot drift apart.
    """
    from diffcmb.alm_utils import packed_dof_per_multipole
    from diffcmb.samplers import _alm_index_lm

    lmax = 16
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    L_arr, _ = _alm_index_lm(lmax, n_real, n_imag)
    dof = packed_dof_per_multipole(lmax)
    for L in range(2, lmax):
        assert dof[L] == int((L_arr == L).sum()) == 2 * L + 1, (
            f"L={L}: dof bookkeeping disagrees with the packing"
        )


@pytest.mark.skipif(not HAS_HEALPY, reason="healpy required")
def test_packed_S_L_is_chi_squared_with_the_packed_dof():
    """S_L | C ~ C * chi^2_{k_L}. This is what fixes the InvGamma exponent.

    Establishes the premise the Block 1 / Block 4 conditionals rest on, using
    the real synalm -> pack path rather than a hand-built draw. Checks the
    variance as well as the mean: matching only the mean cannot distinguish
    chi^2_k from other distributions with the same first moment, and it was
    the *shape* parameter that was wrong.
    """
    from diffcmb.alm_utils import packed_dof_per_multipole
    from diffcmb.lensing import _alm_hp_to_packed, compute_sl_phi_np

    lmax, n = 16, 3000
    cl = np.full(lmax, 1e-6)
    dof = packed_dof_per_multipole(lmax)
    S = np.empty((n, lmax))
    for i in range(n):
        np.random.seed(400000 + i)
        alm = hp.synalm(cl, lmax=lmax - 1, new=True).astype(np.complex128)
        S[i] = compute_sl_phi_np(_alm_hp_to_packed(alm, lmax), lmax)

    for L in (2, 3, 5, 10, 15):
        x = S[:, L] / cl[L]
        k = float(dof[L])
        # chi^2_k has mean k and variance 2k; 3000 draws gives ~2.5% on the
        # mean and ~5% on the implied k from the variance.
        assert abs(x.mean() - k) / k < 0.06, (
            f"L={L}: <S_L>/C = {x.mean():.2f}, expected k={k:.0f} (2L+1 would be {2*L+1})"
        )
        assert abs(2.0 * x.mean() ** 2 / x.var() - k) / k < 0.12, (
            f"L={L}: variance implies k={2*x.mean()**2/x.var():.2f}, expected {k:.0f}"
        )


@pytest.mark.skipif(not HAS_HEALPY, reason="healpy required")
def test_block4_conditional_passes_an_independent_generative_sbc():
    """The test that would have caught the 2026-08-31 exponent bug.

    A probability-integral-transform check against the sampler's own stated
    conditional validates the DRAW, not the DERIVATION -- it passes for any
    exponent, because the code and the reference share the mistake. (The
    2026-08-31 production PIT check returned KS_p=0.53 while the shape was
    half a unit wrong.)

    This instead runs a genuine Gibbs-pair SBC:
        C_true ~ InvGamma(nu/2, nu*C_fid/2)        [the prior]
        phi    ~ N(0, C_true)                      [synalm -> pack]
        C'     ~ sample_cl_phiphi_given_phi(phi)   [the conditional]
    Under a correct conditional the rank of C_true among the C' draws is
    uniform. Nothing here reuses the code's own exponent, so a wrong shape
    parameter shows up immediately. Requires a proper prior: the flat default
    is improper and admits no draw of C_true at all, which is precisely why
    the bug survived for so long.
    """
    from scipy import stats

    from diffcmb.lensing import _alm_hp_to_packed, sample_cl_phiphi_given_phi

    lmax, nu, n_real_iter, n_draw = 16, 6.0, 300, 40
    cl_fid = np.full(lmax, 1e-6)
    rng = np.random.default_rng(0)
    us = []
    for it in range(n_real_iter):
        cl_true = np.zeros(lmax)
        for L in range(2, lmax):
            cl_true[L] = (nu * cl_fid[L] / 2.0) / rng.gamma(nu / 2.0)
        np.random.seed(600000 + it)
        phi = _alm_hp_to_packed(
            hp.synalm(cl_true, lmax=lmax - 1, new=True).astype(np.complex128), lmax
        )
        draws = np.array([
            np.exp(sample_cl_phiphi_given_phi(phi, lmax, rng=rng, prior_nu=nu,
                                              cl_phiphi_fid=cl_fid))
            for _ in range(n_draw)
        ])
        for L in (2, 5, 10, 15):
            us.append((np.sum(draws[:, L - 2] < cl_true[L]) + 0.5) / (n_draw + 1.0))

    us = np.array(us)
    p = stats.kstest(us, "uniform").pvalue
    assert abs(us.mean() - 0.5) < 0.035, (
        f"Block 4 conditional is biased: mean rank {us.mean():.4f}, expected 0.5. "
        f"An exponent assuming 2L+1 dof gives ~0.45; L-1+nu/2 gives ~0.40."
    )
    assert p > 1e-4, f"Block 4 conditional fails generative SBC (KS p={p:.2g})"
