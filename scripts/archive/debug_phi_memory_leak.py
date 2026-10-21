"""
Root-cause investigation for the confirmed phi-block memory leak
(achievements.md, "OPEN, unfixed... ~362 MB/sweep", 2026-07-30).

Hypothesis under test: `lens_map_phi_diff_tf`/`psi_lensed` (lensing.py) define
their `tf.custom_gradient`-wrapped closures (and the `tf.py_function` calls
inside them) as NESTED functions, freshly created on every Python call. The
phi HMC block runs eagerly (not @tf.function-traced -- samplers.py's own
comment at the phi_hmc_one_step call site), so every leapfrog gradient
evaluation re-executes those function bodies in eager mode. Eager-mode
`tf.py_function` calls register their Python callable in a TensorFlow
process-global token registry that is never cleared -- a documented TF
footgun. If this is the leak, RSS should climb roughly linearly with the
number of psi_lensed value+gradient evaluations, independent of lmax/nside
(matching the achievements.md RSS probe's finding that it reproduces at
lmax=96/nside=64), and roughly independent of whether the *result* is used.

This script isolates psi_lensed's value+gradient call (the exact operation
phi_hmc_one_step performs once per leapfrog step) from the rest of the Gibbs
machinery, so a leak here can't be blamed on anything else in run_gibbs_chain.

Usage: OMP_NUM_THREADS=8 TF_ENABLE_ONEDNN_OPTS=0 PYTHONPATH=diffcmb \
    .venv/bin/python scripts/archive/debug_phi_memory_leak.py --n_calls 2000
"""
import argparse
import gc
import time

import numpy as np
import tensorflow as tf

from diffcmb import CosmologyAdvancedSampling
from diffcmb.lensing import psi_lensed


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lmax", type=int, default=32)
    p.add_argument("--nside", type=int, default=32)
    p.add_argument("--n_calls", type=int, default=2000)
    p.add_argument("--report_every", type=int, default=100)
    args = p.parse_args()

    print(f"=== Phi-block memory-leak isolation test: psi_lensed value+grad "
          f"in a bare eager loop (lmax={args.lmax}, nside={args.nside}) ===")
    print("Testing whether repeated eager tf.py_function calls (no HMC "
          "machinery, no Gibbs loop) alone reproduce the leak signature.\n")

    model = CosmologyAdvancedSampling(
        _lmax=args.lmax, _NSIDE=args.nside, _noisesig=1.0,
        data_mode="synthetic", dtype=tf.complex128, use_matrixfree_sht=True,
    )
    model._ensure_tf_tensors()

    # Reuse the model's own x0 (already the right shape/layout) rather than
    # hand-rolling the packed-alm layout here. phi uses the identical packed
    # real/imag layout as the alm portion of x0 (same L,m structure).
    params_tf = tf.constant(model.x0, dtype=tf.float64)
    n_phi = len(model.x0) - (args.lmax - 2)
    rng = np.random.default_rng(0)
    phi_packed = rng.normal(0, 1e-4, size=n_phi)

    print(f"n_params(alm+lncl)={len(model.x0)}, n_phi={n_phi}")
    print(f"Initial RSS: {rss_mb():.1f} MB\n")

    t0 = time.time()
    rss_trace = []
    for i in range(args.n_calls):
        phi_tf = tf.constant(phi_packed, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(phi_tf)
            val = psi_lensed(model, params_tf, phi_tf)
        grad = tape.gradient(val, phi_tf)
        # Touch the outputs so nothing is optimized away, but don't retain them.
        _ = float(val.numpy()) + float(tf.reduce_sum(grad).numpy())
        del tape, val, grad, phi_tf

        if (i + 1) % args.report_every == 0:
            gc.collect()
            r = rss_mb()
            rss_trace.append((i + 1, r))
            elapsed = time.time() - t0
            print(f"  call {i + 1:5d}: RSS={r:8.1f} MB  "
                  f"({elapsed:.1f}s, {elapsed / (i + 1) * 1000:.1f}ms/call)")

    if len(rss_trace) >= 2:
        (n0, r0), (n1, r1) = rss_trace[0], rss_trace[-1]
        rate = (r1 - r0) / (n1 - n0)
        print(f"\nRSS growth rate: {rate * 1024:.2f} KB/call "
              f"(measured from call {n0} to {n1})")
        print(f"Total RSS growth over {n1} calls: {r1 - rss_trace[0][1] + (r0 - 0):.1f} MB "
              f"(first-measured {r0:.1f} -> last {r1:.1f} MB)")
        if rate > 0.05:  # >50 KB/call is well above normal TF/py steady-state noise
            print("VERDICT: leak reproduced in isolation -- psi_lensed's eager "
                  "tf.py_function calls, with NO HMC/Gibbs machinery involved, "
                  "are sufficient to leak memory. Supports the eager "
                  "tf.py_function-registration hypothesis.")
        else:
            print("VERDICT: no significant leak in this isolated loop -- the "
                  "hypothesis is NOT confirmed by this test; leak must involve "
                  "something else in the HMC/Gibbs path (mass matrix rebuild, "
                  "kernel bootstrap, checkpoint accumulation, etc).")


if __name__ == "__main__":
    main()
