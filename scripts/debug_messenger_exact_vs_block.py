"""
Follow-up to debug_messenger_coherent_mask.py / the lmax=60 real-data run:
those showed messenger mixing IS genuinely slow on the real masked-sky
problem (unlike small synthetic toys), and gets dramatically worse from
lmax=60 to lmax=300 -- i.e. this is a dimensionality/scale effect. This
script isolates WHY: is it because the block-diagonal-by-m A^T A
approximation (m_group_size=5, ROADMAP.md Phase 0c Step 5) discards enough
residual cross-block coupling that it compounds into slow mixing at scale,
or is slow mixing intrinsic to messenger sampling here regardless of how
exactly A^T A is captured?

Test: at the SAME cheap lmax=60/NSIDE=64 real-data scale as the previous
script, compare the block-diagonal-by-m correction (m_group_size=5,
production's choice) against the EXACT DENSE A^T A correction
(sample_s_given_t_dense/AtA=..., only tractable at small n_alm, which is
exactly why production had to approximate it in the first place) -- same
starting point, same tau2, same RNG seed pattern. If the dense correction
converges dramatically faster, the block approximation's residual coupling
is the bottleneck (points to increasing m_group_size or switching to
messenger-as-CG-preconditioner). If it's similarly slow, the bottleneck is
intrinsic to messenger Gibbs mixing at this scale (points to tau2
annealing or an entirely different sampler).
"""
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.insert(0, "diffcmb")

from diffcmb import CosmologyAdvancedSampling
from diffcmb.alm_utils import packed_sizes
from diffcmb.messenger import (
    build_block_cholesky,
    run_messenger_gibbs,
    sample_t_given_s,
)
from diffcmb.samplers import (
    _alm_ho_to_packed,
    _alm_index_lm,
    _build_inv_cl_diag,
    _calibrate_block_AtA,
    _packed_to_alm_ho,
)
from diffcmb.sht_ducc import HealpixSHT

LMAX = 60
NSIDE = 64
DATA_DIR = "/cosma8/data/dp004/dc-hick2/Plank"


def main():
    print(f"=== Exact dense vs block-diagonal-by-m A^T A: mixing speed at lmax={LMAX} ===\n")

    model = CosmologyAdvancedSampling(
        _lmax=LMAX, _NSIDE=NSIDE, _noisesig=1.0,
        data_mode="real", data_dir=DATA_DIR,
        dtype=tf.complex128, use_matrixfree_sht=True,
    )
    model._ensure_tf_tensors()

    n_real = LMAX * (LMAX + 1) // 2 - 3
    n_imag = packed_sizes(LMAX)[1]
    n_alm = n_real + n_imag
    L_arr, _m_arr = _alm_index_lm(LMAX, n_real, n_imag)
    print(f"n_alm = {n_alm}")

    rng_init = np.random.default_rng(0)
    lncl_np = np.log(np.full(LMAX - 2, 10.0))
    lncl_full = np.zeros(LMAX)
    lncl_full[2:] = lncl_np
    cl_full = np.exp(lncl_full)
    inv_cl_diag = _build_inv_cl_diag(LMAX, cl_full, n_real, n_imag)

    sht_full = HealpixSHT(nside=NSIDE, lmax=LMAX, unmasked_idx=None,
                           nthreads=getattr(model, "sht_nthreads", 0))

    def A_action(s):
        return sht_full.synthesis_full(_packed_to_alm_ho(s, LMAX, n_real))

    def At_action(t):
        alm_ho = sht_full._w * sht_full.adjoint_synthesis_full(t)
        return _alm_ho_to_packed(alm_ho, LMAX)

    Ninv_full = np.asarray(model.Ninv, dtype=np.float64)
    d_full = np.asarray(model.prior_map, dtype=np.float64)
    Ninv_obs = Ninv_full[Ninv_full > 0]
    tau2 = 0.9 / float(Ninv_obs.max())

    print("Probing EXACT dense A^T A (n_alm basis-vector synthesis calls)...")
    t0 = time.time()
    e = np.zeros(n_alm, dtype=np.float64)
    npix = len(d_full)
    maps = np.empty((n_alm, npix))
    for i in range(n_alm):
        e[:] = 0.0
        e[i] = 1.0
        maps[i] = sht_full.synthesis_full(_packed_to_alm_ho(e, LMAX, n_real))
    AtA_exact = maps @ maps.T
    print(f"  done in {time.time()-t0:.1f}s")

    print("Precomputing dense Cholesky (tau2/inv_cl_diag fixed here, cache "
          "it once rather than recomputing inside sample_s_given_t_dense every call)...")
    t0 = time.time()
    L_dense = np.linalg.cholesky(AtA_exact / tau2 + np.diag(inv_cl_diag))
    print(f"  done in {time.time()-t0:.1f}s\n")

    def sample_s_given_t_dense_cached(At_t, rng):
        y = np.linalg.solve(L_dense, At_t / tau2)
        mean = np.linalg.solve(L_dense.T, y)
        noise = np.linalg.solve(L_dense.T, rng.standard_normal(len(At_t)))
        return mean + noise

    probe_ells = sorted({max(2, min(LMAX - 1, v)) for v in
                          np.linspace(2, LMAX - 1, 7).round().astype(int)})

    def summarize(s):
        out = {"total_energy": float(np.sum(s ** 2))}
        for l in probe_ells:
            mask = L_arr == l
            out[f"l{l}"] = float(np.mean(s[mask] ** 2)) if mask.any() else float("nan")
        return out

    s_far = rng_init.standard_normal(n_alm) * 5.0
    checkpoints = [1, 10, 30, 100, 300, 1000]

    atat_blocks = _calibrate_block_AtA(sht_full, LMAX, n_real, n_imag, m_group_size=5)
    block_chol = build_block_cholesky(atat_blocks, inv_cl_diag, tau2)

    from diffcmb.messenger import sample_s_given_t_block

    for label, s_given_t_fn in [
        ("block (m_group_size=5, production choice)",
         lambda At_t, rng: sample_s_given_t_block(At_t, tau2, rng, block_chol)),
        ("exact dense A^T A",
         sample_s_given_t_dense_cached),
    ]:
        print(f"--- {label} ---")
        rng = np.random.default_rng(42)
        s = s_far.copy()
        n_done = 0

        for target in checkpoints:
            for _ in range(target - n_done):
                s_pix = A_action(s)
                t = sample_t_given_s(s_pix, d_full, Ninv_full, tau2, rng)
                At_t = At_action(t)
                s = s_given_t_fn(At_t, rng)
            n_done = target
            summ = summarize(s)
            print(f"  iter={n_done:5d}  total_energy={summ['total_energy']:.4e}  "
                  + "  ".join(f"l{l}={summ[f'l{l}']:.3e}" for l in probe_ells))
        print()


if __name__ == "__main__":
    main()
