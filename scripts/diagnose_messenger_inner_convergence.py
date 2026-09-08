"""
Phase 0c Step 6 diagnostic: does the messenger sampler's INNER Gibbs loop
(t|s,d / s|t) reach its own equilibrium within n_messenger_iter=100 at
production scale (lmax=300, NSIDE=256, real Planck data)?

ROADMAP.md Phase 0c Step 6 found the production chains (n_messenger_iter=100,
m_group_size=5) have NOT converged after 520 outer Gibbs sweeps: logp/C_l
still drift monotonically, ESS is 0.6% of the Phase 0 HMC baseline. The
leading hypothesis is that each "sample" from sample_alm_messenger is a
partially-relaxed, warm-started inner state rather than a near-exact draw
from p(alm | C_l, d) -- i.e. n_messenger_iter is too small at this scale,
even though it was validated as sufficient at the lmax=16/NSIDE=8 scale
(Step 5).

Method: run the inner messenger Gibbs loop from two very different starting
points (s0=0 and s0=<a real production alm_state from the actual chain>) at
FIXED C_l (the checkpoint's lncl_state), recording a handful of summary
statistics every few iterations. If the inner chain has forgotten its start
and reached equilibrium, both trajectories should agree; if they still
disagree after n_messenger_iter=100 (or more), that confirms insufficient
inner mixing as the root cause -- and shows how many inner iterations are
actually needed, without waiting for a multi-day outer chain.

Uses the exact production settings (block-diagonal-by-m A^T A correction,
m_group_size=5 by default) and reuses the tested library primitives directly
(messenger.sample_t_given_s / sample_s_given_t_block, samplers._calibrate_*)
rather than re-deriving any of the math.

Defaults match production (lmax=300/NSIDE=256/real data); pass --lmax 16
--nside 16 --data_mode synthetic --checkpoint "" for a fast smoke test.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "diffcmb")))

import tensorflow as tf  # noqa: E402

from diffcmb import CosmologyAdvancedSampling  # noqa: E402
from diffcmb.alm_utils import packed_sizes
from diffcmb.messenger import (  # noqa: E402
    build_block_cholesky,
    sample_s_given_t_block,
    sample_t_given_s,
)
from diffcmb.samplers import (  # noqa: E402
    _alm_ho_to_packed,
    _alm_index_lm,
    _build_inv_cl_diag,
    _calibrate_block_AtA,
    _packed_to_alm_ho,
)
from diffcmb.sht_ducc import HealpixSHT  # noqa: E402


def summarize(s, L_arr, probe_ells):
    """Cheap scalar summaries of an alm state: mean(s^2) within each probed l
    (a proxy for that l's contribution to the C_l estimate) plus total energy.
    """
    out = {"total_energy": float(np.sum(s ** 2))}
    for l in probe_ells:
        mask = L_arr == l
        out[f"l{l}"] = float(np.mean(s[mask] ** 2)) if mask.any() else float("nan")
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lmax", type=int, default=300)
    p.add_argument("--nside", type=int, default=256)
    p.add_argument("--data_mode", choices=["real", "synthetic"], default="real")
    p.add_argument("--data_dir", type=str, default="/cosma8/data/dp004/dc-hick2/Plank")
    p.add_argument("--checkpoint", type=str,
                    default="results/lmax300_nside256_gibbs_real_messenger/checkpoint_chain_1.npz",
                    help="Source of lncl_state/alm_state for the 'warm_from_chain' trajectory. "
                         "Pass '' to fall back to a random warm start (smoke-test mode).")
    p.add_argument("--m_group_size", type=int, default=5)
    p.add_argument("--checkpoints", type=int, nargs="+",
                    default=[1, 3, 10, 30, 100, 300, 1000, 3000, 10000])
    p.add_argument("--probe_ells", type=int, nargs="+", default=None,
                    help="Defaults to a spread across [2, lmax-1] if not given.")
    p.add_argument("--out", type=str, default="results/analysis/messenger_inner_convergence_L300.npz")
    return p.parse_args()


def main():
    args = parse_args()
    lmax, nside = args.lmax, args.nside
    probe_ells = args.probe_ells or sorted({
        max(2, min(lmax - 1, v)) for v in
        np.linspace(2, lmax - 1, 7).round().astype(int)
    })

    print(f"=== Messenger inner-loop convergence diagnostic: lmax={lmax}, nside={nside} ===\n")

    print(f"Constructing model (data_mode={args.data_mode}, matrix-free SHT, double precision)...")
    t0 = time.time()
    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=1.0,
        data_mode=args.data_mode, data_dir=args.data_dir if args.data_mode == "real" else None,
        dtype=tf.complex128, use_matrixfree_sht=True,
    )
    model._ensure_tf_tensors()
    print(f"  done in {time.time()-t0:.1f}s\n")

    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    n_alm = n_real + n_imag
    L_arr, _m_arr = _alm_index_lm(lmax, n_real, n_imag)

    rng_init = np.random.default_rng(0)
    if args.checkpoint:
        print(f"Loading chain state from {args.checkpoint}...")
        ckpt = np.load(args.checkpoint)
        lncl_np = ckpt["lncl_state"]
        alm_prod = ckpt["alm_state"]
        assert lncl_np.shape == (lmax - 2,), (lncl_np.shape, lmax - 2)
        assert alm_prod.shape == (n_alm,), (alm_prod.shape, n_alm)
    else:
        print("No checkpoint given -- using a random lncl/alm as the 'warm' start (smoke-test mode).")
        lncl_np = np.log(np.full(lmax - 2, 10.0))
        alm_prod = rng_init.standard_normal(n_alm) * 5.0

    lncl_full = np.zeros(lmax)
    lncl_full[2:] = lncl_np
    cl_full = np.exp(lncl_full)
    inv_cl_diag = _build_inv_cl_diag(lmax, cl_full, n_real, n_imag)

    print("Building full-sky SHT operator...")
    sht_full = HealpixSHT(nside=nside, lmax=lmax, unmasked_idx=None,
                           nthreads=getattr(model, "sht_nthreads", 0))

    def A_action(s):
        return sht_full.synthesis_full(_packed_to_alm_ho(s, lmax, n_real))

    def At_action(t):
        alm_ho = sht_full._w * sht_full.adjoint_synthesis_full(t)
        return _alm_ho_to_packed(alm_ho, lmax)

    Ninv_full = np.asarray(model.Ninv, dtype=np.float64)
    d_full = np.asarray(model.prior_map, dtype=np.float64)
    Ninv_obs = Ninv_full[Ninv_full > 0]
    tau2 = 0.9 / float(Ninv_obs.max())

    print(f"Calibrating block A^T A (m_group_size={args.m_group_size}, {n_alm} modes)...")
    t0 = time.time()
    atat_blocks = _calibrate_block_AtA(
        sht_full, lmax, n_real, n_imag,
        progress_every=10000, m_group_size=args.m_group_size,
    )
    print(f"  done in {(time.time()-t0)/60:.1f} min\n")

    block_chol = build_block_cholesky(atat_blocks, inv_cl_diag, tau2)

    starts = {
        "zero": np.zeros(n_alm, dtype=np.float64),
        "warm_from_chain": alm_prod.astype(np.float64),
    }

    # Run both trajectories in LOCKSTEP (advance each to the next checkpoint
    # in turn, rather than finishing one fully before starting the other) so
    # that the gap comparison -- the actual answer to "has the inner loop
    # converged?" -- is available and saved after every checkpoint, not only
    # once both trajectories finish every target. At production scale each
    # inner iteration costs ~3s (block-Cholesky solves dominate, not the SHT),
    # far more than the ~17ms assumed when picking checkpoints up to 10000 --
    # so a walltime cutoff partway through must still leave usable output.
    state = {name: {"s": s0.copy(), "n_done": 0,
                     "rng": np.random.default_rng(abs(hash(name)) % (2 ** 31))}
              for name, s0 in starts.items()}
    # trajectories[name] = list of (iter_count, wall_seconds, summary_dict)
    trajectories = {name: [] for name in starts}
    t_start = time.time()

    def save_partial():
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        done_checkpoints = [t[0] for t in trajectories["zero"]]
        save_dict = {"checkpoints": np.array(done_checkpoints), "probe_ells": np.array(probe_ells)}
        for name in starts:
            save_dict[f"{name}_iters"] = np.array([t[0] for t in trajectories[name]])
            save_dict[f"{name}_wall_s"] = np.array([t[1] for t in trajectories[name]])
            save_dict[f"{name}_total_energy"] = np.array([t[2]["total_energy"] for t in trajectories[name]])
            for l in probe_ells:
                save_dict[f"{name}_l{l}"] = np.array([t[2][f"l{l}"] for t in trajectories[name]])
        np.savez(args.out, **save_dict)

    print("=== Advancing both trajectories in lockstep ===")
    print("(gap should shrink toward ~0 as iter count grows, if the inner chain mixes)\n")
    for target in args.checkpoints:
        for name in starts:
            st = state[name]
            for _ in range(target - st["n_done"]):
                s_pix = A_action(st["s"])
                t = sample_t_given_s(s_pix, d_full, Ninv_full, tau2, st["rng"])
                At_t = At_action(t)
                st["s"] = sample_s_given_t_block(At_t, tau2, st["rng"], block_chol)
            st["n_done"] = target
            elapsed = time.time() - t_start
            summ = summarize(st["s"], L_arr, probe_ells)
            trajectories[name].append((target, elapsed, summ))

        a = trajectories["zero"][-1][2]
        b = trajectories["warm_from_chain"][-1][2]
        gaps = [abs(a[f"l{l}"] - b[f"l{l}"]) / (0.5 * (abs(a[f"l{l}"]) + abs(b[f"l{l}"])) + 1e-300)
                for l in probe_ells]
        cum_elapsed = time.time() - t_start
        print(f"iter={target:6d}  cum_wall={cum_elapsed:7.1f}s  "
              f"mean rel. gap = {np.mean(gaps):.4f}  max rel. gap = {np.max(gaps):.4f}")
        print(f"    zero:            total_energy={a['total_energy']:.4e}  "
              + "  ".join(f"l{l}={a[f'l{l}']:.3e}" for l in probe_ells))
        print(f"    warm_from_chain: total_energy={b['total_energy']:.4e}  "
              + "  ".join(f"l{l}={b[f'l{l}']:.3e}" for l in probe_ells))

        save_partial()  # keep the on-disk result usable even if the job is killed mid-run

    print(f"\nSaved trajectories (updated after every checkpoint) -> {args.out}")
    print("Done.")


if __name__ == "__main__":
    main()
