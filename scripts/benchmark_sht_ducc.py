"""Phase 1.5 gate benchmark: matrix-free ducc0 SHT vs the dense `sph` matrix.

ROADMAP.md gate: forward+backward at lmax=300, NSIDE=256 must come in
under ~1s (vs the measured 9.38s dense GPU+CPU path, scripts/benchmark_lensing.py,
job 11552544). Run on a CPU-only node — no dense matrix, no GPU needed.

Run with:
  python scripts/benchmark_sht_ducc.py
"""
import sys
import time

import numpy as np

sys.path.insert(0, '/cosma/apps/durham/dc-hick2/diffcmb/diffcmb')

import healpy as hp
import tensorflow as tf

from diffcmb.sht_ducc import HealpixSHT, masked_synthesis_tf

DATA_DIR = '/cosma8/data/dp004/dc-hick2/Plank'
LMAX = 300
NSIDE = 256
NTHREADS = 8


def main():
    mask_file = f"{DATA_DIR}/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
    raw_mask = hp.read_map(mask_file, field=0)
    mask = hp.ud_grade(raw_mask, nside_out=NSIDE)
    unmasked_idx = np.where(mask > 0.9)[0]
    print(f"lmax={LMAX} nside={NSIDE} unmasked={len(unmasked_idx)}/{hp.nside2npix(NSIDE)}")

    sht = HealpixSHT(nside=NSIDE, lmax=LMAX, unmasked_idx=unmasked_idx, nthreads=NTHREADS)
    print(f"n_alm={sht.n_alm}")

    rng = np.random.default_rng(0)
    alm = rng.standard_normal(sht.n_alm) + 1j * rng.standard_normal(sht.n_alm)
    _, ms = hp.Alm.getlm(LMAX - 1, i=np.arange(sht.n_alm))
    alm[ms == 0] = alm[ms == 0].real
    alm_tf = tf.constant(alm.astype(np.complex128), dtype=tf.complex128)
    weights = tf.constant(rng.standard_normal(len(unmasked_idx)), dtype=tf.float64)

    # Warm-up (thread pool / first-call overhead).
    with tf.GradientTape() as tape:
        tape.watch(alm_tf)
        out = masked_synthesis_tf(alm_tf, sht)
        loss = tf.reduce_sum(out * weights)
    tape.gradient(loss, alm_tf)

    t0 = time.time()
    with tf.GradientTape() as tape:
        tape.watch(alm_tf)
        out = masked_synthesis_tf(alm_tf, sht)
        loss = tf.reduce_sum(out * weights)
    t1 = time.time()
    grad = tape.gradient(loss, alm_tf)
    t2 = time.time()

    print(f"forward:  {t1 - t0:.4f}s")
    print(f"backward: {t2 - t1:.4f}s")
    print(f"total:    {t2 - t0:.4f}s")
    print(f"gate (<1s): {'PASS' if (t2 - t0) < 1.0 else 'FAIL'}")
    print(f"grad shape/dtype: {grad.shape} {grad.dtype}")


if __name__ == '__main__':
    main()
