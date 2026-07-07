"""ROADMAP.md Phase 0c Step 6, follow-up to benchmark_messenger_block_lmax300.py:
that script only isolated the block-Cholesky *rebuild* cost, not the full
per-outer-sweep sample_alm_messenger cost -- which also pays n_messenger_iter
forward/adjoint full-sky SHTs (A_action/At_action) each sweep, plus (with
use_block_correction=True) a per-block triangular solve every inner
iteration, not just at rebuild time. This times the actual end-to-end call
production chains will make, at lmax=300 with real Planck data, for the
m_group_size candidates the isolated-cost benchmark showed were cheaper than
the 6.7s/PCG-iteration CG baseline (1, 3, 5).

One-time calibration costs (_calibrate_full_sky_norm_diag, _calibrate_block_AtA)
are paid once per model per m_group_size and cached on the model instance --
timed separately here since they must not be counted against the recurring
per-sweep cost.

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/benchmark_messenger_fullcall_lmax300.py
"""
import gc
import os
import time

import numpy as np

from diffcmb.model import CosmologyAdvancedSampling
from diffcmb.samplers import sample_alm_messenger

LMAX = 300
NSIDE = 256
NOISE = 1.0
DATA_DIR = "/cosma8/data/dp004/dc-hick2/Plank"
N_MESSENGER_ITER = 100  # run_sampler.py / run_gibbs_chain default
N_TIMED_CALLS = 5
# job 11562981 (2026-07-04) OOM-killed under --mem=32G while calibrating
# m_group_size=5: _calibrate_block_AtA builds one (block_size, npix) dense
# array per block, and near m=0 that's ~2930 x 786432 float64 (~18GB)
# transiently, on top of the already-cached m_group_size=1/3 AtA blocks.
# Rerun only the untested group size here (1, 3 already timed: 89.62s,
# 188.08s mean per sweep); drop each group's cache after timing it so
# peak memory doesn't grow with the number of group sizes tested.
M_GROUP_SIZES = tuple(
    int(x) for x in os.environ.get("MSGR_M_GROUP_SIZES", "5").split(",")
)
CG_BASELINE_SEC_PER_ITER = 6.7  # ROADMAP.md Phase 0c reference point


def main():
    print(f"Building model: lmax={LMAX}, NSIDE={NSIDE}, real Planck data...")
    model = CosmologyAdvancedSampling(
        LMAX, NSIDE, NOISE, data_mode='real', data_dir=DATA_DIR, dtype=None,
        use_matrixfree_sht=True,
    )
    model._ensure_tf_tensors()

    lmax = model.lmax
    lncl_np = np.log(np.maximum(model.prior_cls[2:lmax], 1e-8))
    rng = np.random.default_rng(0)

    print("\nWarming diagonal-approx calibration cache (one-time, shared across "
          "all m_group_size runs below)...")
    t0 = time.perf_counter()
    s = sample_alm_messenger(model, lncl_np, rng, n_messenger_iter=1)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    for m_group_size in M_GROUP_SIZES:
        print(f"\n=== m_group_size={m_group_size} ===")
        t0 = time.perf_counter()
        s = sample_alm_messenger(
            model, lncl_np, rng, n_messenger_iter=N_MESSENGER_ITER, s0=s,
            use_block_correction=True, m_group_size=m_group_size,
        )
        calib_and_first_call_sec = time.perf_counter() - t0
        print(f"  one-time block-AtA calibration + first full sweep: "
              f"{calib_and_first_call_sec:.1f}s")

        print(f"  timing {N_TIMED_CALLS} subsequent full sweeps "
              f"(n_messenger_iter={N_MESSENGER_ITER}, calibration cached)...")
        times = []
        for _ in range(N_TIMED_CALLS):
            t0 = time.perf_counter()
            s = sample_alm_messenger(
                model, lncl_np, rng, n_messenger_iter=N_MESSENGER_ITER, s0=s,
                use_block_correction=True, m_group_size=m_group_size,
            )
            times.append(time.perf_counter() - t0)
        times = np.array(times)
        print(f"  per-sweep wall time: mean={times.mean():.2f}s, "
              f"min={times.min():.2f}s, max={times.max():.2f}s")
        print(f"  vs CG baseline ({CG_BASELINE_SEC_PER_ITER}s/PCG-iteration): "
              f"{'CHEAPER' if times.mean() < CG_BASELINE_SEC_PER_ITER else 'MORE EXPENSIVE'} "
              f"per unit (note: PCG never converged on this problem at any "
              f"iteration budget, so this comparison is cost-only, not accuracy-adjusted)")

        # Drop this group size's cached AtA blocks before moving to the next
        # one -- each block's calibration transiently allocates a
        # (block_size, npix) dense array (~18GB for m_group_size=5's largest
        # near-m=0 block), and letting multiple group sizes' caches pile up
        # on the model is what caused job 11562981's OOM kill.
        model._messenger_AtA_blocks_cache.pop(m_group_size, None)
        gc.collect()


if __name__ == "__main__":
    main()
