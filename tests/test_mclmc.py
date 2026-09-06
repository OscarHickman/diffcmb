"""MCLMC integrator — invariant and finiteness tests (spike for Block 3).

Validation strategy (mirrors tests/test_lensing.py's dense-reference discipline):
1. isokinetic_momentum_update / partially_refresh_momentum preserve ||u||=1
   exactly — cheap, reference-free checks that catch basic implementation bugs
   immediately, before any comparison against the real potential.
2. mclachlan_step / mclmc_trajectory produce finite output for finite input,
   using the real log_prob_phi_block potential at the same small lmax=20 scale
   tests/test_lensing.py uses.
"""

import numpy as np
import pytest

from diffcmb.alm_utils import packed_sizes

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import healpy as hp
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False

LMAX = 20
NSIDE = 16


def _make_model(lmax=LMAX, nside=NSIDE):
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


def _rand_unit_vector(n, rng, dtype=None):
    if dtype is None:
        dtype = tf.float64
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    return tf.constant(v, dtype=dtype)


# ---------------------------------------------------------------------------
# 1 — invariants, no model/potential needed
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")
def test_isokinetic_momentum_update_preserves_unit_norm():
    from diffcmb.mclmc import isokinetic_momentum_update
    rng = np.random.default_rng(0)
    n = 50
    u = _rand_unit_vector(n, rng)
    grad = tf.constant(rng.standard_normal(n) * 3.0, dtype=tf.float64)
    u_new, ke = isokinetic_momentum_update(u, grad, step_size=tf.constant(0.1, tf.float64), coef=tf.constant(0.5, tf.float64))
    norm = tf.linalg.norm(u_new).numpy()
    assert np.isclose(norm, 1.0, atol=1e-12)
    assert np.isfinite(ke.numpy())


@pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")
def test_isokinetic_momentum_update_handles_zero_gradient():
    """A zero gradient must not produce NaN (division-by-zero guard)."""
    from diffcmb.mclmc import isokinetic_momentum_update
    rng = np.random.default_rng(1)
    n = 20
    u = _rand_unit_vector(n, rng)
    grad = tf.zeros(n, dtype=tf.float64)
    u_new, ke = isokinetic_momentum_update(u, grad, step_size=tf.constant(0.1, tf.float64), coef=tf.constant(0.5, tf.float64))
    assert np.all(np.isfinite(u_new.numpy()))
    assert np.isclose(tf.linalg.norm(u_new).numpy(), 1.0, atol=1e-12)


@pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")
def test_partially_refresh_momentum_preserves_unit_norm():
    from diffcmb.mclmc import partially_refresh_momentum
    rng = np.random.default_rng(2)
    n = 50
    u = _rand_unit_vector(n, rng)
    tf.random.set_seed(42)
    u_new = partially_refresh_momentum(u, L=tf.constant(5.0, tf.float64), step_size=tf.constant(0.05, tf.float64))
    assert np.isclose(tf.linalg.norm(u_new).numpy(), 1.0, atol=1e-12)


@pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")
def test_partially_refresh_momentum_large_L_barely_moves():
    """L -> large (slow decoherence) should leave u close to unchanged."""
    from diffcmb.mclmc import partially_refresh_momentum
    rng = np.random.default_rng(3)
    n = 50
    u = _rand_unit_vector(n, rng)
    tf.random.set_seed(7)
    u_new = partially_refresh_momentum(u, L=tf.constant(1e6, tf.float64), step_size=tf.constant(0.01, tf.float64))
    cos_sim = tf.tensordot(u, u_new, axes=1).numpy()
    assert cos_sim > 0.999


# ---------------------------------------------------------------------------
# 2 — finiteness against the real Block 3 potential (lmax=20, matches
#     tests/test_lensing.py's dense-reference scale)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (HAS_TF and HAS_HEALPY), reason="tensorflow/healpy not installed")
def test_mclachlan_step_finite_on_real_potential():
    from diffcmb.lensing import log_prob_phi_block
    from diffcmb.mclmc import mclachlan_step

    model = _make_model()
    lmax = model.lmax
    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    n_phi = n_real + n_imag

    rng = np.random.default_rng(11)
    params_np = np.zeros(n_lncl + n_real + n_imag)
    params_np[:n_lncl] = 5.0
    params_np[n_lncl:] = rng.standard_normal(n_real + n_imag) * 0.1
    params_tf = tf.constant(params_np, dtype=tf.float64)
    cl_phiphi = np.full(lmax, 1e-8)

    x = tf.constant(rng.standard_normal(n_phi) * 1e-5, dtype=tf.float64)
    u = _rand_unit_vector(n_phi, rng)

    def grad_fn(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            lp = log_prob_phi_block(model, params_tf, x, cl_phiphi)
        return lp, tape.gradient(lp, x)

    x_new, u_new, ke = mclachlan_step(x, u, grad_fn, step_size=tf.constant(1e-6, tf.float64))
    assert np.all(np.isfinite(x_new.numpy()))
    assert np.all(np.isfinite(u_new.numpy()))
    assert np.isclose(tf.linalg.norm(u_new).numpy(), 1.0, atol=1e-10)
    assert np.isfinite(ke.numpy())


@pytest.mark.skipif(not (HAS_TF and HAS_HEALPY), reason="tensorflow/healpy not installed")
def test_mclmc_trajectory_finite_and_moves():
    from diffcmb.lensing import log_prob_phi_block
    from diffcmb.mclmc import mclmc_trajectory

    model = _make_model()
    lmax = model.lmax
    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    n_phi = n_real + n_imag

    rng = np.random.default_rng(23)
    params_np = np.zeros(n_lncl + n_real + n_imag)
    params_np[:n_lncl] = 5.0
    params_np[n_lncl:] = rng.standard_normal(n_real + n_imag) * 0.1
    params_tf = tf.constant(params_np, dtype=tf.float64)
    cl_phiphi = np.full(lmax, 1e-8)

    x = tf.constant(rng.standard_normal(n_phi) * 1e-5, dtype=tf.float64)
    u = _rand_unit_vector(n_phi, rng)

    def grad_fn(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            lp = log_prob_phi_block(model, params_tf, x, cl_phiphi)
        return lp, tape.gradient(lp, x)

    tf.random.set_seed(99)
    x_new, u_new, diag = mclmc_trajectory(
        x, u, grad_fn, step_size=tf.constant(1e-6, tf.float64), L=tf.constant(5.0, tf.float64), n_steps=5
    )
    assert np.all(np.isfinite(x_new.numpy()))
    assert np.all(np.isfinite(u_new.numpy()))
    assert np.isclose(tf.linalg.norm(u_new).numpy(), 1.0, atol=1e-9)
    assert np.isfinite(diag.energy_error.numpy())
    assert not diag.diverged.numpy()
    assert not np.allclose(x_new.numpy(), x.numpy())
