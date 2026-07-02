"""Dense vs matrix-free SHT agreement in the actual model (Phase 1.5).

ROADMAP.md calls for validating the matrix-free ducc0 backend against the
dense `sph`-matrix path "before Phase 2 production chains are submitted".
`tests/test_sht_ducc.py` validates `sht_ducc.py` in isolation; this file
validates it wired into `CosmologyAdvancedSampling.psi_tf` (`use_matrixfree_sht`
flag, model.py) — same posterior value and same alm-gradient, dense vs
matrix-free, at a small lmax on synthetic data so it runs in seconds.
"""
import numpy as np
import pytest


def _has_deps():
    try:
        import ducc0  # noqa: F401
        import healpy  # noqa: F401
        import scipy  # noqa: F401
        import tensorflow as tf  # noqa: F401

        from diffcmb import CosmologyAdvancedSampling  # noqa: F401
        return True
    except Exception:
        return False


skip_no_deps = pytest.mark.skipif(not _has_deps(), reason="heavy deps (ducc0/healpy/scipy/tf) unavailable")

LMAX = 12
NSIDE = 16


def _make_model_pair():
    """One synthetic sky (fixed by np's global RNG state at construction time),
    with both a dense-path and a matrix-free-path model built from IDENTICAL
    underlying prior_map/Ninv/unmasked_idx — otherwise `data_mode='synthetic'`
    draws a fresh random sky per instance and the two models aren't comparable."""
    import tensorflow as tf

    from diffcmb.model import CosmologyAdvancedSampling
    from diffcmb.sht_ducc import HealpixSHT

    np.random.seed(0)
    model_dense = CosmologyAdvancedSampling(
        _lmax=LMAX, _NSIDE=NSIDE, _noisesig=1.0, data_mode='synthetic',
        parameterization='centered', dtype=tf.complex128,
        use_matrixfree_sht=False,
    )
    model_dense._ensure_tf_tensors()

    model_free = CosmologyAdvancedSampling(
        _lmax=LMAX, _NSIDE=NSIDE, _noisesig=1.0, data_mode='synthetic',
        parameterization='centered', dtype=tf.complex128,
        use_matrixfree_sht=True, sht_nthreads=2,
    )
    # Overwrite with the dense model's data before building TF tensors, so
    # both models share the exact same synthetic sky/mask/noise.
    model_free.prior_map = model_dense.prior_map.copy()
    model_free.Ninv = model_dense.Ninv.copy()
    model_free.unmasked_idx = model_dense.unmasked_idx.copy()
    model_free._ensure_tf_tensors()

    return model_dense, model_free


@skip_no_deps
def test_psi_tf_value_matches_dense():
    import tensorflow as tf

    model_dense, model_free = _make_model_pair()

    n_real = LMAX * (LMAX + 1) // 2 - 3
    n_imag = (LMAX - 2) * (LMAX - 1) // 2
    n_params = (LMAX - 2) + n_real + n_imag
    rng = np.random.default_rng(0)
    params_np = rng.standard_normal(n_params) * 0.05
    params_np[: LMAX - 2] += 5.0  # plausible lncl scale
    params_tf = tf.constant(params_np, dtype=tf.float64)

    val_dense = float(model_dense._psi_tf_raw(params_tf).numpy())
    val_free = float(model_free._psi_tf_raw(params_tf).numpy())

    assert abs(val_dense - val_free) / abs(val_dense) < 1e-8


@skip_no_deps
def test_psi_tf_grad_matches_dense():
    import tensorflow as tf

    model_dense, model_free = _make_model_pair()

    n_real = LMAX * (LMAX + 1) // 2 - 3
    n_imag = (LMAX - 2) * (LMAX - 1) // 2
    n_params = (LMAX - 2) + n_real + n_imag
    rng = np.random.default_rng(1)
    params_np = rng.standard_normal(n_params) * 0.05
    params_np[: LMAX - 2] += 5.0

    def grad_for(model):
        p = tf.Variable(params_np, dtype=tf.float64)
        with tf.GradientTape() as tape:
            val = model._psi_tf_raw(p)
        return tape.gradient(val, p).numpy()

    g_dense = grad_for(model_dense)
    g_free = grad_for(model_free)

    np.testing.assert_allclose(g_free, g_dense, rtol=1e-6, atol=1e-6)


@skip_no_deps
def test_psi_tf_raw_traceable_with_matrixfree_sht():
    """The matrix-free path must survive tf.function tracing (tf.py_function
    escape hatch), matching how samplers.py wraps _psi_tf_raw in practice."""
    import tensorflow as tf

    _, model = _make_model_pair()
    n_real = LMAX * (LMAX + 1) // 2 - 3
    n_imag = (LMAX - 2) * (LMAX - 1) // 2
    n_params = (LMAX - 2) + n_real + n_imag
    params_np = np.zeros(n_params)
    params_np[: LMAX - 2] = 5.0

    @tf.function(jit_compile=False)
    def _grad_fn(params_tf):
        with tf.GradientTape() as tape:
            tape.watch(params_tf)
            val = model._psi_tf_raw(params_tf)
        return tape.gradient(val, params_tf)

    out = _grad_fn(tf.constant(params_np, dtype=tf.float64))
    assert out.shape == (n_params,)
    assert np.all(np.isfinite(out.numpy()))
