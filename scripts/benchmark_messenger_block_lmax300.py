"""ROADMAP.md Phase 0c Step 6: benchmark use_block_correction=True's per-sweep
cost at production scale (lmax=300, NSIDE=256, real Planck data), the item
Step 5 left open ("Not yet done: benchmarking use_block_correction=True at
production lmax=300 ... vs the 6.7s/PCG-iteration CG baseline").

Two costs matter, and they scale differently:

1. _calibrate_block_AtA: O(n_alm) full-sky SHT synthesis calls, ONE-TIME per
   model (cached on model._messenger_AtA_blocks_cache), independent of
   m_group_size (grouping changes how the n_alm probe vectors are binned
   into blocks, not how many there are).
2. build_block_cholesky: rebuilt every outer Gibbs sweep (depends on the
   current C_l draw) — O(sum_m block_size(m)^3), which DOES grow with
   m_group_size since blocks get wider.

This script times a representative sample of each (rather than the full
one-time calibration, which at n_alm ~ 90k full-sky SHTs would take far
longer than a login-node budget allows) and extrapolates the totals.

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/benchmark_messenger_block_lmax300.py
"""
import time

import numpy as np

from diffcmb.messenger import build_block_cholesky
from diffcmb.model import CosmologyAdvancedSampling
from diffcmb.samplers import _alm_index_lm, _build_inv_cl_diag, _packed_to_alm_ho
from diffcmb.sht_ducc import HealpixSHT

LMAX = 300
NSIDE = 256
NOISE = 1.0
DATA_DIR = "/cosma8/data/dp004/dc-hick2/Plank"
N_PROBE_SAMPLE = 200  # basis vectors actually timed, out of n_alm total
CG_BASELINE_SEC_PER_ITER = 6.7  # ROADMAP.md Phase 0c reference point


def main():
    print(f"Building model: lmax={LMAX}, NSIDE={NSIDE}, real Planck data...")
    model = CosmologyAdvancedSampling(
        LMAX, NSIDE, NOISE, data_mode='real', data_dir=DATA_DIR, dtype=None,
        use_matrixfree_sht=True,
    )
    model._ensure_tf_tensors()

    lmax = model.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2
    n_alm = n_real + n_imag
    print(f"n_alm = {n_alm}")

    sht_full = HealpixSHT(
        nside=model.NSIDE, lmax=lmax, unmasked_idx=None,
        nthreads=getattr(model, "sht_nthreads", 0),
    )

    # --- Cost 1: per-probe full-sky synthesis cost (drives one-time calibration) ---
    print(f"\nTiming {N_PROBE_SAMPLE} full-sky synthesis_full calls "
          f"(basis vectors sampled uniformly across all {n_alm} alm dof)...")
    rng = np.random.default_rng(0)
    probe_idx = rng.choice(n_alm, size=N_PROBE_SAMPLE, replace=False)
    e = np.zeros(n_alm, dtype=np.float64)
    t0 = time.perf_counter()
    for i in probe_idx:
        e[:] = 0.0
        e[i] = 1.0
        alm_ho = _packed_to_alm_ho(e, lmax, n_real)
        sht_full.synthesis_full(alm_ho)
    t1 = time.perf_counter()
    sec_per_probe = (t1 - t0) / N_PROBE_SAMPLE
    total_calib_sec = sec_per_probe * n_alm
    print(f"  {sec_per_probe * 1e3:.3f} ms/probe -> full one-time calibration "
          f"estimate: {total_calib_sec:.1f}s ({total_calib_sec / 60:.1f} min)")

    # --- Cost 2: per-sweep block Cholesky rebuild cost, vs m_group_size ---
    L_arr, m_arr = _alm_index_lm(lmax, n_real, n_imag)
    unique_m = np.unique(m_arr)

    lncl_np = np.log(np.maximum(model.prior_cls[2:lmax], 1e-8))
    lncl_full = np.zeros(lmax)
    lncl_full[2:] = lncl_np
    cl_full = np.exp(lncl_full)
    inv_cl_diag = _build_inv_cl_diag(lmax, cl_full, n_real, n_imag)
    tau2 = 1.0  # arbitrary fixed value; only affects conditioning, not timing

    print("\nTiming build_block_cholesky rebuild cost per outer-sweep, "
          "for representative m_group_size values (using SYNTHETIC AtA blocks "
          "of the correct sizes -- calibrating the real blocks is Cost 1 above, "
          "already amortized as a one-time cost, so this isolates the "
          "per-sweep-only cholesky-rebuild cost):")
    for m_group_size in (1, 3, 5, 10, 20):
        blocks = []
        for g in range(0, len(unique_m), m_group_size):
            m_group = unique_m[g:g + m_group_size]
            idx = np.where(np.isin(m_arr, m_group))[0]
            block_size = len(idx)
            # Synthetic SPD matrix with the right shape/conditioning order --
            # real AtA blocks (Cost 1) don't affect Cholesky wall-time, only
            # block_size does, so this avoids re-running Cost 1 per group size.
            M = rng.standard_normal((block_size, block_size))
            AtA_block = M @ M.T + block_size * np.eye(block_size)
            blocks.append((idx, AtA_block))
        t0 = time.perf_counter()
        build_block_cholesky(blocks, inv_cl_diag, tau2)
        t1 = time.perf_counter()
        max_block = max(len(idx) for idx, _ in blocks)
        print(f"  m_group_size={m_group_size:2d}: {len(blocks)} blocks, "
              f"max block size={max_block}, rebuild time={t1 - t0:.3f}s/sweep")

    print(f"\nCG baseline for reference: {CG_BASELINE_SEC_PER_ITER}s/PCG-iteration "
          f"(ROADMAP.md Phase 0c) -- typical PCG runs did not converge within "
          f"any tractable iteration budget on the masked sky, which is the "
          f"reason for this messenger-field replacement in the first place.")


if __name__ == "__main__":
    main()
