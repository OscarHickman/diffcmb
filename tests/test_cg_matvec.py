"""Regression test for the CG alm|C_l sampler's matvec linearity/symmetry/PD.

p(alm | C_l, d) is exactly Gaussian, so A p := grad(psi)(p) - grad(psi)(0)
must be linear, symmetric and positive-definite for the PCG solver in
sample_alm_cg (samplers.py) to converge.

Bug history: matvec_on_device's tf.custom_gradient left grad_x on the
sph_part's own device, which silently broke TF's cross-GPU gradient
accumulation when sph_parts were split across more than one GPU (production
CG run 11513133 spun for 3 days without the PCG residual ever decreasing).
Fixed in model.py::matvec_on_device by moving grad_x to a common device
before returning. The dense-path test below forces a multi-GPU split (when
>1 GPU is visible) to exercise that exact cross-device path cheaply, without
needing an --exclusive multi-GPU node for every CI run.

On a login node / CPU-only / single-GPU runner, this still validates
linearity and symmetry of A, just without exercising the cross-device
accumulation bug specifically (see `multi_gpu` skip reason below).

Both the dense `sph`-matrix path and the matrix-free ducc0 SHT path
(`use_matrixfree_sht=True`, sht_ducc.py) are checked here (Phase 1.5,
ROADMAP.md): the fresh full-scale `debug_cg.py` run that confirmed A's
symmetry/PD/linearity at lmax=300 used the dense path only, so this
parametrization is the matrix-free path's first A-operator check.
"""
import numpy as np
import pytest

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

DATA_DIR = '/cosma8/data/dp004/dc-hick2/Plank'
LMAX = 30
NSIDE = 256
NOISE = 1.0
N_PARTS = 5


def _force_multi_gpu_split(model, n_parts, gpus):
    """Re-split the model's single sph_part across all visible GPUs so the
    cross-device gradient accumulation path is actually exercised."""
    assert len(model.sph_parts) == 1
    sph_full = model.sph_parts[0]
    prior_map_full = model.prior_map_parts[0]
    Ninv_full = model.Ninv_parts[0]
    npix = int(sph_full.shape[0])
    bounds = np.linspace(0, npix, n_parts + 1).astype(int)

    new_sph_parts, new_prior_parts, new_ninv_parts = [], [], []
    for i in range(n_parts):
        s, e = bounds[i], bounds[i + 1]
        dev = f'/GPU:{i % len(gpus)}'
        with tf.device(dev):
            new_sph_parts.append(tf.identity(sph_full[s:e]))
            new_prior_parts.append(tf.identity(prior_map_full[s:e]))
            new_ninv_parts.append(tf.identity(Ninv_full[s:e]))

    model.sph_parts = new_sph_parts
    model.prior_map_parts = new_prior_parts
    model.Ninv_parts = new_ninv_parts
    model.multi_device = True
    for attr in ("_compiled_psi_tf", "_cg_grad_fn", "_cg_jt_v_fn"):
        if hasattr(model, attr):
            delattr(model, attr)


def _check_matvec_linear_symmetric_pd(model):
    lmax = model.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2
    n_alm = n_real + n_imag
    lncl_np = np.log(model.prior_cls[2:LMAX] + 1e-30)
    lncl_tf_c = tf.constant(lncl_np, dtype=tf.float64)

    @tf.function(jit_compile=False)
    def _grad_fn(lncl_tf, alm_tf):
        with tf.GradientTape() as tape:
            tape.watch(alm_tf)
            full_params = tf.concat([lncl_tf, alm_tf], axis=0)
            val = model._psi_tf_raw(full_params)
        return tape.gradient(val, alm_tf)

    def alm_grad(alm_np):
        return _grad_fn(lncl_tf_c, tf.constant(alm_np, dtype=tf.float64)).numpy()

    zeros = np.zeros(n_alm, dtype=np.float64)
    minus_b_data = alm_grad(zeros)

    rng = np.random.default_rng(42)
    p_test = rng.standard_normal(n_alm) * 0.01
    q_test = rng.standard_normal(n_alm) * 0.01

    Ap_1x = alm_grad(p_test) - minus_b_data
    Ap_2x = alm_grad(2 * p_test) - minus_b_data
    Ap_neg = alm_grad(-p_test) - minus_b_data
    Aq = alm_grad(q_test) - minus_b_data

    ratio_2x = np.linalg.norm(Ap_2x) / np.linalg.norm(Ap_1x)
    ratio_neg = np.linalg.norm(Ap_neg) / np.linalg.norm(Ap_1x)
    symmetry_err = (
        abs(np.dot(p_test, Aq) - np.dot(q_test, Ap_1x))
        / (abs(np.dot(p_test, Aq)) + 1e-30)
    )
    pAp = np.dot(p_test, Ap_1x)
    qAq = np.dot(q_test, Aq)

    assert abs(ratio_2x - 2.0) < 1e-3, f"A is not linear: ||A(2p)||/||A(p)||={ratio_2x}"
    assert abs(ratio_neg - 1.0) < 1e-3, f"A is not linear: ||A(-p)||/||A(p)||={ratio_neg}"
    assert symmetry_err < 1e-6, f"A is not symmetric: err={symmetry_err}"
    assert pAp > 0, f"A is not positive-definite: dot(p, Ap)={pAp}"
    assert qAq > 0, f"A is not positive-definite: dot(q, Aq)={qAq}"


@pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")
def test_cg_matvec_linear_symmetric():
    from diffcmb.model import CosmologyAdvancedSampling

    gpus = tf.config.list_physical_devices('GPU')

    model = CosmologyAdvancedSampling(
        _lmax=LMAX, _NSIDE=NSIDE, _noisesig=NOISE, data_mode='real',
        data_dir=DATA_DIR, parameterization='centered', dtype=tf.complex128,
    )
    model._ensure_tf_tensors()

    if len(gpus) > 1:
        _force_multi_gpu_split(model, n_parts=N_PARTS, gpus=gpus)

    _check_matvec_linear_symmetric_pd(model)


@pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")
def test_cg_matvec_linear_symmetric_matrixfree_sht():
    """Same A-operator checks as test_cg_matvec_linear_symmetric, but with
    the matrix-free ducc0 SHT backend (use_matrixfree_sht=True) instead of
    the dense `sph` matrix. Phase 1.5 (ROADMAP.md) calls for this explicitly:
    prior validation of the matrix-free path (test_sht_ducc*.py) checked
    psi_tf value/gradient agreement with the dense path, and a fresh
    debug_cg.py run checked A's symmetry/PD/linearity at production scale —
    but that run used the dense path, not use_matrixfree_sht=True."""
    try:
        import ducc0  # noqa: F401
    except ImportError:
        pytest.skip("ducc0 not installed")

    from diffcmb.model import CosmologyAdvancedSampling

    model = CosmologyAdvancedSampling(
        _lmax=LMAX, _NSIDE=NSIDE, _noisesig=NOISE, data_mode='real',
        data_dir=DATA_DIR, parameterization='centered', dtype=tf.complex128,
        use_matrixfree_sht=True, sht_nthreads=2,
    )
    model._ensure_tf_tensors()

    _check_matvec_linear_symmetric_pd(model)
