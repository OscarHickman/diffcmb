"""Cheap regression check for the CG alm|C_l sampler's matvec (Wandelt+2004).

p(alm | C_l, d) is exactly Gaussian, so A p := grad(psi)(p) - grad(psi)(0)
must be linear, symmetric and positive-definite for the PCG solver in
sample_alm_cg (samplers.py) to converge. Bug history: matvec_on_device's
tf.custom_gradient left grad_x on the sph_part's own device, which silently
broke TF's cross-GPU gradient accumulation when sph_parts were split across
more than one GPU (production CG run 11513133 spun for 3 days without the
PCG residual ever decreasing). Fixed by moving grad_x to a common device
before returning (model.py, matvec_on_device).

Uses real Planck data at a small lmax so the dense sph matrix fits in a
single GPU's memory, then manually re-splits it across all visible GPUs to
exercise the cross-device gradient path cheaply (no --exclusive node needed).
"""
import sys

import numpy as np

sys.path.insert(0, '/cosma/apps/durham/dc-hick2/diffcmb/diffcmb')

import tensorflow as tf

from diffcmb.model import CosmologyAdvancedSampling

DATA_DIR = '/cosma8/data/dp004/dc-hick2/Plank'
LMAX = 30
NSIDE = 256
NOISE = 1.0
N_PARTS = 5


def main():
    gpus = tf.config.list_physical_devices('GPU')
    print("=" * 65)
    print(f"verify_cg_matvec: lmax={LMAX}, {N_PARTS} parts, GPUs={gpus}")
    print("=" * 65)

    model = CosmologyAdvancedSampling(
        _lmax=LMAX, _NSIDE=NSIDE, _noisesig=NOISE, data_mode='real',
        data_dir=DATA_DIR, parameterization='centered', dtype=tf.complex128,
    )
    model._ensure_tf_tensors()

    if len(gpus) > 1:
        _force_multi_gpu_split(model, n_parts=N_PARTS, gpus=gpus)
    else:
        print(f"Only {len(gpus)} GPU(s) visible; running single-device check only.")

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
    symmetry_err = abs(np.dot(p_test, Aq) - np.dot(q_test, Ap_1x)) / (abs(np.dot(p_test, Aq)) + 1e-30)

    print(f"||A(2p)||/||A(p)|| = {ratio_2x:.6f}  (expect 2.0)")
    print(f"||A(-p)||/||A(p)|| = {ratio_neg:.6f}  (expect 1.0)")
    print(f"symmetry_err       = {symmetry_err:.3e}  (expect ~0)")

    assert abs(ratio_2x - 2.0) < 1e-3, f"A is not linear: ratio_2x={ratio_2x}"
    assert abs(ratio_neg - 1.0) < 1e-3, f"A is not linear: ratio_neg={ratio_neg}"
    assert symmetry_err < 1e-6, f"A is not symmetric: err={symmetry_err}"
    print("PASS: CG matvec is linear and symmetric.")


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


if __name__ == '__main__':
    main()
