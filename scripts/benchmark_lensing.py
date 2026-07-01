"""Benchmark forward + backward pass time for psi_lensed at lmax=300, NSIDE=256.

Phase 1 roadmap item: the lensed likelihood (diffcmb/lensing.py::psi_lensed) has
passing gradient-vs-FD tests at lmax=50 but no timing numbers at production
scale. This measures wall-clock cost of one forward pass and one
forward+backward (gradient) pass w.r.t. both alm and phi_alm, since Phase 2's
HMC needs both gradients every leapfrog step.

Uses synthetic data (data_mode="synthetic") since only timing is of interest,
not scientific correctness — avoids needing the real Planck data path.
"""
import sys
import time

import numpy as np

sys.path.insert(0, '/cosma/apps/durham/dc-hick2/diffcmb/diffcmb')

import tensorflow as tf

from diffcmb.lensing import _alm_hp_to_packed, psi_lensed
from diffcmb.model import CosmologyAdvancedSampling

LMAX = 300
NSIDE = 256
N_WARMUP = 3
N_TIMED = 10


def _rand_phi_packed(lmax, rng, amplitude=5e-4):
    import healpy as hp
    size = hp.Alm.getsize(lmax)
    phi_hp = (rng.standard_normal(size) + 1j * rng.standard_normal(size)) * amplitude
    ells = np.array([hp.Alm.getlm(lmax, i)[0] for i in range(size)], dtype=float)
    if lmax >= 2:
        phi_hp[ells < 2] = 0.0
    return _alm_hp_to_packed(phi_hp.astype(np.complex128), lmax)


def main():
    gpus = tf.config.list_physical_devices('GPU')
    print("=" * 70)
    print(f"benchmark_lensing: lmax={LMAX}, NSIDE={NSIDE}, GPUs={gpus}")
    print("=" * 70)

    rng = np.random.default_rng(0)

    t0 = time.time()
    model = CosmologyAdvancedSampling(
        _lmax=LMAX, _NSIDE=NSIDE, _noisesig=100.0,
        data_mode="synthetic", dtype=tf.complex128,
    )
    model._ensure_tf_tensors()
    print(f"Model setup: {time.time() - t0:.1f}s")

    n_real = LMAX * (LMAX + 1) // 2 - 3
    n_imag = (LMAX - 2) * (LMAX - 1) // 2
    n_alm = n_real + n_imag

    lncl_np = np.log(model.prior_cls[2:LMAX] + 1e-30)
    alm_np = rng.standard_normal(n_alm) * 10.0
    params_np = np.concatenate([lncl_np, alm_np])
    phi_packed_np = _rand_phi_packed(LMAX, rng)

    params_tf = tf.constant(params_np, dtype=tf.float64)
    phi_tf = tf.constant(phi_packed_np, dtype=tf.float64)

    def _forward():
        return psi_lensed(model, params_tf, phi_tf)

    def _forward_backward():
        with tf.GradientTape() as tape:
            tape.watch(params_tf)
            tape.watch(phi_tf)
            val = psi_lensed(model, params_tf, phi_tf)
        grads = tape.gradient(val, [params_tf, phi_tf])
        return val, grads

    print(f"\nWarming up ({N_WARMUP} iters each)...")
    for _ in range(N_WARMUP):
        _forward()
    for _ in range(N_WARMUP):
        _forward_backward()

    print(f"\nTiming forward pass ({N_TIMED} iters)...")
    t0 = time.time()
    for _ in range(N_TIMED):
        val = _forward()
    _ = val.numpy()  # force sync
    fwd_time = (time.time() - t0) / N_TIMED
    print(f"  Forward pass:            {fwd_time * 1000:.1f} ms/iter")

    print(f"\nTiming forward + backward pass ({N_TIMED} iters)...")
    t0 = time.time()
    for _ in range(N_TIMED):
        val, grads = _forward_backward()
    _ = [g.numpy() for g in grads]  # force sync
    fwdbwd_time = (time.time() - t0) / N_TIMED
    print(f"  Forward + backward pass: {fwdbwd_time * 1000:.1f} ms/iter")
    print(f"  Backward-only estimate:  {(fwdbwd_time - fwd_time) * 1000:.1f} ms/iter")
    print(f"  Backward/forward ratio:  {fwdbwd_time / fwd_time:.2f}x")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"lmax={LMAX}, NSIDE={NSIDE}, n_alm={n_alm}, n_phi={n_alm}")
    print(f"forward:          {fwd_time * 1000:.1f} ms")
    print(f"forward+backward: {fwdbwd_time * 1000:.1f} ms")
    print(f"Est. leapfrog steps/sec (Phase 2 HMC, fwd+bwd dominated): {1.0 / fwdbwd_time:.2f}")


if __name__ == '__main__':
    main()
