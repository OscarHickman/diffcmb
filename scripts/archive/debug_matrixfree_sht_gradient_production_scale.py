"""
Follow-up to debug_matrixfree_sht_accuracy_production_scale.py: the raw
forward SHT matches healpy to 5.7e-13 at lmax=300/NSIDE=256 (production
scale), ruling out a bug in the forward transform. HMC relies entirely on
psi_tf's GRADIENT (masked_synthesis_tf's tf.custom_gradient backward pass)
for its leapfrog dynamics -- if that gradient has a subtle bug that only
appears at this scale (unlike the lmax=15 test in tests/test_sht_ducc.py),
HMC would systematically be pushed in the wrong direction even though
forward evaluations and the ESS/acceptance diagnostics all look healthy.

Checks the custom gradient against finite differences at PRODUCTION
lmax=300/NSIDE=256, real Planck mask -- a handful of probed alm coefficients
(cheap: each FD check costs 2 extra forward passes at ~0.018s each).
"""
import healpy as hp
import numpy as np
import tensorflow as tf

from diffcmb.sht_ducc import HealpixSHT, masked_synthesis_tf

DATA_DIR = "/cosma8/data/dp004/dc-hick2/Plank"
LMAX = 300
NSIDE = 256


def main():
    mask_file = f"{DATA_DIR}/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
    raw_mask = hp.read_map(mask_file, field=0)
    mask = hp.ud_grade(raw_mask, nside_out=NSIDE)
    unmasked_idx = np.where(mask > 0.9)[0]

    sht = HealpixSHT(nside=NSIDE, lmax=LMAX, unmasked_idx=unmasked_idx, nthreads=8)
    print(f"n_alm={sht.n_alm}, n_unmasked={len(unmasked_idx)}")

    rng = np.random.default_rng(0)
    alm0 = rng.standard_normal(sht.n_alm) + 1j * rng.standard_normal(sht.n_alm)
    _, ms = hp.Alm.getlm(LMAX - 1, i=np.arange(sht.n_alm))
    alm0[ms == 0] = alm0[ms == 0].real
    alm0 = alm0.astype(np.complex128)

    weights = rng.standard_normal(len(unmasked_idx))

    def loss_np(alm_np):
        m = sht.masked_synthesis(alm_np)
        return float(np.sum(m * weights))

    # Analytic gradient via the actual production tf.custom_gradient path.
    alm_tf = tf.constant(alm0, dtype=tf.complex128)
    weights_tf = tf.constant(weights, dtype=tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(alm_tf)
        out = masked_synthesis_tf(alm_tf, sht)
        loss = tf.reduce_sum(out * weights_tf)
    grad_analytic = tape.gradient(loss, alm_tf).numpy()

    print(f"loss = {float(loss):.6e}")
    print("\nFinite-difference check on a handful of probed alm indices "
          "(real and imaginary perturbations separately):")
    probe_indices = [0, 1, 100, sht.n_alm // 4, sht.n_alm // 2,
                      sht.n_alm - 100, sht.n_alm - 1]
    eps = 1e-5
    max_rel_err = 0.0
    for i in probe_indices:
        # d(loss)/d(Re(alm_i))
        alm_p = alm0.copy()
        alm_p[i] += eps
        alm_m = alm0.copy()
        alm_m[i] -= eps
        fd_re = (loss_np(alm_p) - loss_np(alm_m)) / (2 * eps)

        alm_p = alm0.copy()
        alm_p[i] += 1j * eps
        alm_m = alm0.copy()
        alm_m[i] -= 1j * eps
        fd_im = (loss_np(alm_p) - loss_np(alm_m)) / (2 * eps)

        g = grad_analytic[i]
        rel_err_re = abs(g.real - fd_re) / (abs(fd_re) + 1e-300)
        rel_err_im = abs(g.imag - fd_im) / (abs(fd_im) + 1e-300)
        max_rel_err = max(max_rel_err, rel_err_re, rel_err_im)
        print(f"  i={i:6d}  analytic=({g.real:.4e}, {g.imag:.4e})j  "
              f"FD=({fd_re:.4e}, {fd_im:.4e})j  rel_err=({rel_err_re:.2e}, {rel_err_im:.2e})")

    print(f"\nmax relative error across probed indices: {max_rel_err:.3e}")
    print("PASS" if max_rel_err < 1e-3 else "FAIL — gradient disagrees with finite differences")


if __name__ == "__main__":
    main()
