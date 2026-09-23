"""Exact lensing operator (lensing.lens_alm_exact_tf) -- value and gradient checks.

The exact operator replaces bilinear interpolation for production (ROADMAP D3):
it evaluates the band-limited unlensed alm at the deflected positions with
ducc0's non-uniform SHT. What must hold:

1. value: phi=0 reproduces the ordinary synthesis; at nonzero phi it agrees
   with a brute-force sum_lm a_lm Y_lm(theta', phi') at the deflected angles.
2. the gradient fields it uses for d/dphi (dT/dtheta, dT/dphi at the deflected
   point) agree with finite differences of the value.
3. d psi_lensed / d alm and d psi_lensed / d phi agree with finite
   differences, through the whole model (this pins the complex-gradient and
   m>0-weight conventions, which are easy to get wrong by a conjugate or 2).
4. the remapping is geodesic (n' = exp_n(grad phi)); at small deflection the
   bilinear operator's first-order coordinate offset converges to it as the
   interpolation grid is refined.
5. wiring: lens_map_tf / model options / tf.function tracing.
"""

import numpy as np
import pytest

hp = pytest.importorskip("healpy")
tf = pytest.importorskip("tensorflow")
pytest.importorskip("ducc0")

from diffcmb.alm_utils import packed_length, packed_sizes  # noqa: E402
from diffcmb.lensing import (  # noqa: E402
    DEFLECTION_SIGN_LEGACY,
    DEFLECTION_SIGN_PHYSICAL,
    _alm_hp_to_packed,
    _alm_packed_to_hp,
    deflection_field,
    exact_lens_np,
    lens_alm_exact_tf,
)

LMAX, NSIDE = 16, 16


def _rand_alm_hp(lmax, rng, scale=10.0):
    size = hp.Alm.getsize(lmax - 1)
    a = (rng.standard_normal(size) + 1j * rng.standard_normal(size)) * scale
    ells, ms = hp.Alm.getlm(lmax - 1)
    a[ms == 0] = a[ms == 0].real
    a[ells < 2] = 0.0
    return a


def _rand_phi_packed(lmax, rng, amplitude=3e-3):
    size = hp.Alm.getsize(lmax - 1)
    ells, ms = hp.Alm.getlm(lmax - 1)
    phi = (rng.standard_normal(size) + 1j * rng.standard_normal(size)) * amplitude
    phi /= np.maximum(ells, 1.0) ** 1.5
    phi[ms == 0] = phi[ms == 0].real
    phi[ells < 2] = 0.0
    return _alm_hp_to_packed(phi, lmax)


def _model(lmax=LMAX, nside=NSIDE, **kw):
    from diffcmb import CosmologyAdvancedSampling

    m = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=100.0, data_mode="synthetic",
        dtype=tf.complex128, use_matrixfree_sht=True, **kw)
    m._ensure_tf_tensors()
    return m


# ---------------------------------------------------------------------------
# 1. value
# ---------------------------------------------------------------------------

def test_exact_lensing_is_plain_synthesis_at_zero_phi():
    rng = np.random.default_rng(0)
    alm = _rand_alm_hp(LMAX, rng)
    pix = np.arange(hp.nside2npix(NSIDE))
    T = exact_lens_np(alm, np.zeros(packed_length(LMAX)), NSIDE, LMAX, pix)
    np.testing.assert_allclose(T, hp.alm2map(alm, NSIDE, lmax=LMAX - 1), atol=1e-8)


def _rodrigues_deflect(theta, phi, a, b):
    """Independent geodesic remap: rotate n about axis n x d_hat by |d|."""
    n = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], 1)
    e_th = np.stack([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)], 1)
    e_ph = np.stack([-np.sin(phi), np.cos(phi), 0 * phi], 1)
    d = a[:, None] * e_th + b[:, None] * e_ph
    alpha = np.linalg.norm(d, axis=1)
    k = np.cross(n, d / alpha[:, None])
    k /= np.linalg.norm(k, axis=1, keepdims=True)
    out = (n * np.cos(alpha)[:, None] + np.cross(k, n) * np.sin(alpha)[:, None]
           + k * np.sum(k * n, axis=1, keepdims=True) * (1 - np.cos(alpha))[:, None])
    return np.arccos(np.clip(out[:, 2], -1, 1)), np.mod(np.arctan2(out[:, 1], out[:, 0]), 2 * np.pi)


def test_exact_lensing_matches_brute_force_harmonic_sum():
    """Value at the geodesically deflected points, against scipy's Y_lm and an
    independent (Rodrigues-rotation) construction of those points."""
    from scipy.special import sph_harm

    rng = np.random.default_rng(1)
    lmax = 8
    alm = _rand_alm_hp(lmax, rng)
    phi = _rand_phi_packed(lmax, rng, amplitude=0.05)     # large, visible deflections
    pix = rng.choice(hp.nside2npix(NSIDE), size=40, replace=False)
    T = exact_lens_np(alm, phi, NSIDE, lmax, pix)

    d_th, d_ph = deflection_field(_alm_packed_to_hp(phi, lmax), NSIDE, lmax,
                                  sign=DEFLECTION_SIGN_PHYSICAL)
    th0, ph0 = hp.pix2ang(NSIDE, pix)
    th, ph = _rodrigues_deflect(th0, ph0, d_th[pix], d_ph[pix] * np.sin(th0))
    ref = np.zeros(len(pix))
    for ell in range(lmax):
        for m in range(ell + 1):
            a = alm[hp.Alm.getidx(lmax - 1, ell, m)]
            y = sph_harm(m, ell, ph, th)            # scipy: (m, l, azimuth, polar)
            ref += (a * y).real if m == 0 else 2.0 * (a * y).real
    np.testing.assert_allclose(T, ref, atol=1e-8)
    assert np.max(np.abs(np.hypot(d_th, d_ph))) > 1e-2   # the test actually lenses


def test_geodesic_map_moves_points_by_exactly_the_deflection_length():
    from diffcmb.lensing import _exact_lensing_geometry

    rng = np.random.default_rng(11)
    lmax = 12
    phi = _rand_phi_packed(lmax, rng, amplitude=0.2)      # degree-scale deflections
    pix = np.arange(hp.nside2npix(NSIDE))
    loc = _exact_lensing_geometry(phi, NSIDE, lmax, pix)
    d_th, d_ph = deflection_field(_alm_packed_to_hp(phi, lmax), NSIDE, lmax,
                                  sign=DEFLECTION_SIGN_PHYSICAL)
    th0, ph0 = hp.pix2ang(NSIDE, pix)
    moved = hp.rotator.angdist(np.stack([th0, ph0]), loc.T)
    np.testing.assert_allclose(moved, np.hypot(d_th, d_ph * np.sin(th0)), atol=1e-9)
    assert np.max(moved) > 0.05


def test_deflection_sign_conventions_against_healpy_gradient():
    """PHYSICAL is +grad(phi); LEGACY (every bilinear chain) is -grad(phi).

    Found 2026-09-23: the original deflection_field used glm = -sqrt(l(l+1))
    phi_lm, which healpy's spin-1 synthesis turns into -grad(phi).
    """
    rng = np.random.default_rng(10)
    nside, lmax = 32, 16
    phi = _alm_packed_to_hp(_rand_phi_packed(lmax, rng), lmax)
    _, dth_ref, dph_over_sin_ref = hp.alm2map_der1(phi, nside, lmax=lmax - 1)
    th, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    for sign in (DEFLECTION_SIGN_PHYSICAL, DEFLECTION_SIGN_LEGACY):
        dth, dph = deflection_field(phi, nside, lmax, sign=sign)
        np.testing.assert_allclose(dth, sign * dth_ref, atol=1e-10)
        np.testing.assert_allclose(dph * np.sin(th), sign * dph_over_sin_ref, atol=1e-10)
    assert DEFLECTION_SIGN_PHYSICAL == 1.0 and DEFLECTION_SIGN_LEGACY == -1.0


# ---------------------------------------------------------------------------
# 2. gradient fields
# ---------------------------------------------------------------------------

def test_deflection_derivatives_match_finite_differences():
    """dT/d(d_theta), dT/d(d_phi) through the geodesic map, against FD of an
    independent evaluation (Rodrigues remap + synthesis_general)."""
    import ducc0

    rng = np.random.default_rng(2)
    alm = _rand_alm_hp(LMAX, rng)
    phi = _rand_phi_packed(LMAX, rng, amplitude=0.05)
    pix = rng.choice(hp.nside2npix(NSIDE), size=60, replace=False)
    _, _, dT_da, dT_ddphi = exact_lens_np(alm, phi, NSIDE, LMAX, pix, with_gradient=True)

    d_th, d_ph = deflection_field(_alm_packed_to_hp(phi, LMAX), NSIDE, LMAX,
                                  sign=DEFLECTION_SIGN_PHYSICAL)
    th0, ph0 = hp.pix2ang(NSIDE, pix)
    a0, dphi0 = d_th[pix], d_ph[pix]

    def T_of(a, dphi):
        th, ph = _rodrigues_deflect(th0, ph0, a, dphi * np.sin(th0))
        loc = np.ascontiguousarray(np.stack([th, ph], 1))
        return ducc0.sht.synthesis_general(alm=alm[None], spin=0, lmax=LMAX - 1,
                                           loc=loc, epsilon=1e-12)[0]

    eps = 1e-6
    fd_a = (T_of(a0 + eps, dphi0) - T_of(a0 - eps, dphi0)) / (2 * eps)
    fd_p = (T_of(a0, dphi0 + eps) - T_of(a0, dphi0 - eps)) / (2 * eps)
    np.testing.assert_allclose(dT_da, fd_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dT_ddphi, fd_p, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. gradients through psi_lensed / log_prob_phi_block
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def exact_model():
    return _model(lensing_operator="exact")


def _params(lmax, rng):
    n_lncl = lmax - 2
    p = np.zeros(n_lncl + packed_length(lmax))
    p[:n_lncl] = 5.0
    p[n_lncl:] = rng.standard_normal(packed_length(lmax)) * 0.3
    return p


def _fd(f, x, idx, eps):
    out = []
    for i in idx:
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        out.append((f(xp) - f(xm)) / (2 * eps))
    return np.array(out)


def test_psi_lensed_exact_alm_gradient_vs_fd(exact_model):
    from diffcmb.lensing import psi_lensed

    rng = np.random.default_rng(3)
    lmax = exact_model.lmax
    params = _params(lmax, rng)
    phi = tf.constant(_rand_phi_packed(lmax, rng), tf.float64)
    var = tf.Variable(params, dtype=tf.float64)
    with tf.GradientTape() as tape:
        val = psi_lensed(exact_model, var, phi)
    g = tape.gradient(val, var).numpy()

    n_lncl = lmax - 2
    n_real, _ = packed_sizes(lmax)
    # lnC_l, real parts (m=0 and m>0) and imaginary parts
    idx = [0, 5, n_lncl, n_lncl + 1, n_lncl + 7, n_lncl + n_real - 1,
           n_lncl + n_real, n_lncl + n_real + 9, len(params) - 1]
    fd = _fd(lambda x: psi_lensed(exact_model, tf.constant(x, tf.float64), phi).numpy(),
             params, idx, 1e-5)
    np.testing.assert_allclose(g[idx], fd, rtol=1e-5, atol=1e-6)


def test_psi_lensed_exact_phi_gradient_vs_fd(exact_model):
    """No bilinear cell boundaries: the exact operator is smooth in phi, so
    an ordinary FD step works (the bilinear test needed eps=1e-9)."""
    from diffcmb.lensing import psi_lensed

    rng = np.random.default_rng(4)
    lmax = exact_model.lmax
    params = tf.constant(_params(lmax, rng), tf.float64)
    phi = _rand_phi_packed(lmax, rng)
    var = tf.Variable(phi, dtype=tf.float64)
    with tf.GradientTape() as tape:
        val = psi_lensed(exact_model, params, var)
    g = tape.gradient(val, var).numpy()

    n_real, _ = packed_sizes(lmax)
    idx = [0, 1, 4, 10, n_real - 1, n_real, n_real + 5, len(phi) - 1]
    fd = _fd(lambda x: psi_lensed(exact_model, params, tf.constant(x, tf.float64)).numpy(),
             phi, idx, 1e-7)
    np.testing.assert_allclose(g[idx], fd, rtol=1e-4, atol=1e-5)


def test_log_prob_phi_block_exact_gradient_vs_fd(exact_model):
    from diffcmb.lensing import log_prob_phi_block

    rng = np.random.default_rng(5)
    lmax = exact_model.lmax
    params = tf.constant(_params(lmax, rng), tf.float64)
    phi = _rand_phi_packed(lmax, rng)
    clpp = np.full(lmax, 1e-6)
    var = tf.Variable(phi, dtype=tf.float64)
    with tf.GradientTape() as tape:
        val = log_prob_phi_block(exact_model, params, var, clpp)
    g = tape.gradient(val, var).numpy()
    idx = [2, 17, len(phi) - 3]
    fd = _fd(lambda x: log_prob_phi_block(exact_model, params,
                                          tf.constant(x, tf.float64), clpp).numpy(),
             phi, idx, 1e-7)
    np.testing.assert_allclose(g[idx], fd, rtol=1e-4, atol=1e-5)


def test_exact_psi_lensed_is_tf_function_traceable(exact_model):
    from diffcmb.lensing import psi_lensed

    rng = np.random.default_rng(6)
    lmax = exact_model.lmax
    params = tf.constant(_params(lmax, rng), tf.float64)
    phi = tf.Variable(_rand_phi_packed(lmax, rng), dtype=tf.float64)

    @tf.function
    def value_and_grad():
        with tf.GradientTape() as tape:
            v = psi_lensed(exact_model, params, phi)
        return v, tape.gradient(v, phi)

    v, g = value_and_grad()
    assert np.isfinite(v.numpy()) and np.all(np.isfinite(g.numpy()))
    np.testing.assert_allclose(v.numpy(), psi_lensed(exact_model, params, phi).numpy(),
                               rtol=1e-12)


# ---------------------------------------------------------------------------
# 4. same displacement convention as the bilinear operator
# ---------------------------------------------------------------------------

def test_bilinear_converges_to_exact_as_the_grid_is_refined():
    from diffcmb.lensing import precompute_lensing

    rng = np.random.default_rng(7)
    lmax = 12
    alm = _rand_alm_hp(lmax, rng)
    phi = _rand_phi_packed(lmax, rng)
    errs = []
    for nside in (16, 32, 64):
        pix = np.arange(hp.nside2npix(nside))
        exact = exact_lens_np(alm, phi, nside, lmax, pix)
        # The bilinear path keeps the legacy -grad(phi) sign, so it lenses by
        # +grad(phi) when handed -phi.
        nb, w, _, _ = precompute_lensing(_alm_packed_to_hp(-phi, lmax), nside, lmax, pix)
        bil = np.sum(w * hp.alm2map(alm, nside, lmax=lmax - 1)[nb], axis=0)
        errs.append(np.sqrt(np.mean((bil - exact) ** 2)))
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 0.3 * errs[0]


# ---------------------------------------------------------------------------
# 5. wiring
# ---------------------------------------------------------------------------

def test_lens_map_tf_uses_the_exact_operator(exact_model):
    from diffcmb.lensing import lens_map_tf

    rng = np.random.default_rng(8)
    lmax = exact_model.lmax
    alm_packed = rng.standard_normal(packed_length(lmax)) * 5.0
    phi_packed = _rand_phi_packed(lmax, rng)
    got = lens_map_tf(exact_model, tf.constant(alm_packed, tf.float64),
                      _alm_packed_to_hp(phi_packed, lmax)).numpy()
    want = exact_lens_np(_alm_packed_to_hp(alm_packed, lmax), phi_packed,
                         exact_model.NSIDE, lmax, exact_model.unmasked_idx)
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_exact_operator_requires_matrixfree_and_a_known_name():
    from diffcmb import CosmologyAdvancedSampling

    with pytest.raises(ValueError, match="matrixfree"):
        CosmologyAdvancedSampling(_lmax=8, _NSIDE=8, _noisesig=1.0,
                                  lensing_operator="exact", use_matrixfree_sht=False)
    with pytest.raises(ValueError, match="lensing_operator"):
        CosmologyAdvancedSampling(_lmax=8, _NSIDE=8, _noisesig=1.0,
                                  lensing_operator="cubic", use_matrixfree_sht=True)


def test_lens_alm_exact_tf_value_matches_numpy():
    rng = np.random.default_rng(9)
    alm = _rand_alm_hp(LMAX, rng)
    phi = _rand_phi_packed(LMAX, rng)
    pix = np.arange(hp.nside2npix(NSIDE))
    got = lens_alm_exact_tf(tf.constant(alm, tf.complex128), tf.constant(phi, tf.float64),
                            NSIDE, LMAX, pix).numpy()
    np.testing.assert_allclose(got, exact_lens_np(alm, phi, NSIDE, LMAX, pix), atol=1e-12)
