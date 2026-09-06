"""
Fresh-hypothesis diagnosis (ROADMAP.md NEXT SESSION, 2026-08-17): every
tuning-axis attempt at the phi-block equilibration prerequisite (2 samplers x
up to 3 scales x 2 mass-matrix variants x 3 further tuning axes -- see
achievements.md's "Coverage-test prerequisite findings") has failed. All of
those preconditioners (prior-only, one-shot Fisher) are *diagonal in L* --
they rescale each packed-phi coordinate independently but assume no
coupling between coordinates in different L-bins.

lensing.py::estimate_phi_diag_fisher's own docstring already records a
strong hint this assumption is wrong: an earlier joint-direction (Hutchinson)
curvature estimator was abandoned because "the off-diagonal coupling in this
Hessian turned out large enough that single-digit probe counts gave
estimates off by orders of magnitude" -- but that finding was never
quantified or connected to the equilibration failures in achievements.md.
This script does that directly: it measures actual cross-L Hessian-vector
coupling (not just an abandoned-estimator anecdote) using the exact same
validated finite-difference-of-analytic-gradient method
estimate_phi_diag_fisher already uses for the diagonal, extended to
off-diagonal blocks.

Method: reconstruct the model+data for an already-completed pilot chain
(default: job 11759467, lmax=64/nside=64/seed=0) bit-for-bit (same LCDM_PARAMS,
same seed streams as pilot_coverage_equilibration.py), then at several phi
states sampled along that chain:

  1. Diagonal per-L curvature (estimate_phi_diag_fisher) -- for direct
     comparison against the numbers already recorded in achievements.md's
     "Fisher-mass-matrix geometry fix" entry, and to sanity check this
     script's own setup reproduces the model correctly.
  2. Off-diagonal coupling: for a source coordinate j in one L-bin, perturb
     phi_packed[j] by eps and take the analytic-gradient finite difference
     (same eps=1e-9, same single-coordinate perturbation style as
     estimate_phi_diag_fisher -- deliberately NOT a joint multi-coordinate
     probe, to avoid the exact bilinear-interpolation-cell-boundary
     instability that abandoned the Hutchinson estimator). Compare the
     resulting gradient response within j's own L-bin (the diagonal-ish
     entries) against the response in a *different* L-bin (the coupling
     entries). The ratio ||cross-bin response|| / ||within-bin response||
     quantifies how much of the curvature a diagonal-in-L mass matrix
     (prior or Fisher, whichever) necessarily discards.

No new MCMC. Cheap: O(n_probes * n_bin_pairs) single-coordinate gradient
evaluations, each one lensing forward+backward pass at lmax=64 (~seconds
each per the pilot's own per-sweep timings).

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/diagnose_phi_hessian_coupling.py \
    --chain_npz results/analysis/pilot_coverage_lmax64_hmc_extended_n6600.npz \
    --lmax 64 --nside 64 --n_chain_points 4 --n_probes 6
"""
import argparse

import healpy as hp
import numpy as np
import tensorflow as tf
from pilot_coverage_equilibration import LCDM_PARAMS, get_cl_phiphi

from diffcmb import CosmologyAdvancedSampling
from diffcmb.alm_utils import packed_sizes
from diffcmb.lensing import estimate_phi_diag_fisher, psi_lensed
from diffcmb.power import call_CAMB_map
from diffcmb.samplers import _alm_index_lm


def rebuild_model_and_data(lmax, nside, noisesig, seed):
    """Bit-for-bit reproduction of pilot_coverage_equilibration.py's model
    and noisy-map construction (same seed streams), so the loaded chain's
    phi/alm samples are probed against the exact likelihood they were
    actually sampled under."""
    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=noisesig,
        use_matrixfree_sht=True, sht_nthreads=8,
    )
    model._ensure_tf_tensors()

    cl_true = call_CAMB_map(LCDM_PARAMS, lmax)
    cl_phiphi_true = get_cl_phiphi(lmax)

    rng_truth = np.random.default_rng(seed)
    rng_noise = np.random.default_rng(seed + 20_000)

    np.random.seed(rng_truth.integers(0, 2**31 - 1))
    alm_true_hp = hp.synalm(cl_true, lmax=lmax - 1, new=True).astype(np.complex128)
    phi_true_hp = hp.synalm(cl_phiphi_true, lmax=lmax - 1, new=True).astype(np.complex128)
    from diffcmb.lensing import _alm_hp_to_packed, lens_map_tf

    # lens_map_tf expects alm in *packed* (author-ordered real/imag) layout
    # but phi in raw healpy complex-alm layout -- matches
    # pilot_coverage_equilibration.py's truth-lensing call exactly.
    alm_true_packed = _alm_hp_to_packed(alm_true_hp, lmax)
    T_lensed_true = lens_map_tf(
        model, tf.constant(alm_true_packed, tf.float64), phi_true_hp
    ).numpy()
    noisy_map = T_lensed_true + rng_noise.normal(0.0, noisesig, size=model.NPIX)
    model.prior_map = noisy_map
    model.prior_map_masked = tf.convert_to_tensor(
        noisy_map[model.unmasked_idx], dtype=tf.float64
    )

    return model, cl_true, cl_phiphi_true


def analytic_grad(model, params_tf, phi_val):
    phi_var = tf.Variable(phi_val, dtype=tf.float64)
    with tf.GradientTape() as tape:
        val = psi_lensed(model, params_tf, phi_var)
    return tape.gradient(val, phi_var).numpy()


def cross_bin_coupling(model, params_tf, phi_np, L_arr, bin_j, bin_i, n_probes, rng, eps=1e-9):
    """For n_probes random source coordinates j in bin_j, perturb phi[j] and
    measure the analytic-gradient finite-difference response magnitude
    within bin_j itself (diagonal-ish) vs within bin_i (cross-bin coupling).
    """
    idx_j = np.where((L_arr >= bin_j[0]) & (L_arr < bin_j[1]))[0]
    idx_i = np.where((L_arr >= bin_i[0]) & (L_arr < bin_i[1]))[0]
    if len(idx_j) == 0 or len(idx_i) == 0:
        return np.nan, np.nan, 0

    g0 = analytic_grad(model, params_tf, phi_np)
    probe_j = rng.choice(idx_j, size=min(n_probes, len(idx_j)), replace=False)

    within_mags, cross_mags = [], []
    for j in probe_j:
        phi_pert = phi_np.copy()
        phi_pert[j] += eps
        g1 = analytic_grad(model, params_tf, phi_pert)
        dgrad = (g1 - g0) / eps
        within_mags.append(np.abs(dgrad[idx_j]).mean())
        cross_mags.append(np.abs(dgrad[idx_i]).mean())

    return float(np.mean(within_mags)), float(np.mean(cross_mags)), len(probe_j)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chain_npz", type=str,
                   default="results/analysis/pilot_coverage_lmax64_hmc_extended_n6600.npz")
    p.add_argument("--lmax", type=int, default=64)
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--noisesig", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_chain_points", type=int, default=4)
    p.add_argument("--n_probes", type=int, default=6)
    p.add_argument("--fisher_probes", type=int, default=8)
    p.add_argument("--out", type=str,
                   default="results/analysis/diagnose_phi_hessian_coupling.npz")
    args = p.parse_args()

    print(f"=== Reconstructing model+data: lmax={args.lmax}, nside={args.nside}, "
          f"seed={args.seed} (matching {args.chain_npz}) ===")
    model, cl_true, cl_phiphi_true = rebuild_model_and_data(
        args.lmax, args.nside, args.noisesig, args.seed
    )

    d = np.load(args.chain_npz)
    phi_samples = d["phi_samples"]
    alm_samples = d["alm_samples"]
    n_total = phi_samples.shape[0]

    n_real = args.lmax * (args.lmax + 1) // 2 - 3
    n_imag = packed_sizes(args.lmax)[1]
    L_arr, _ = _alm_index_lm(args.lmax, n_real, n_imag)

    bins = [(2, 10), (10, 30), (30, 60), (60, args.lmax)]
    stuck_bin = (60, args.lmax)  # the bin that fails the equilibration gate at lmax=64
    healthy_bin = (10, 30)  # cleanly equilibrates in every lmax=64 pilot so far

    rng = np.random.default_rng(12345)
    chain_idx = np.linspace(n_total // 4, n_total - 1, args.n_chain_points, dtype=int)

    print(f"\n=== Diagonal per-L Fisher curvature at {args.n_chain_points} chain points ===")
    diag_records = []
    for ci in chain_idx:
        # alm_samples[ci] already has the full x0 layout [lncl(lmax-2),
        # real_alm, imag_alm] -- same as psi_lensed's params_tf -- so it's
        # used directly, not re-concatenated with a separate lncl.
        params_tf = tf.constant(alm_samples[ci], dtype=tf.float64)
        phi_np = phi_samples[ci].astype(np.float64)
        diag_fisher = estimate_phi_diag_fisher(
            model, params_tf, tf.constant(phi_np, dtype=tf.float64),
            args.lmax, n_probes=args.fisher_probes, rng=rng,
        )
        row = []
        for (lo, hi) in bins:
            vals = diag_fisher[lo:hi]
            vals = vals[vals > 0]
            row.append(float(np.mean(vals)) if len(vals) else float("nan"))
        diag_records.append(row)
        print(f"  chain point {ci:5d}: " +
              "  ".join(f"[{lo},{hi})={v:.3e}" for (lo, hi), v in zip(bins, row)))

    print(f"\n=== Cross-L Hessian coupling: stuck bin {stuck_bin} <-> healthy bin {healthy_bin} ===")
    print("  (within-bin = diagonal-ish response; cross-bin = coupling a diagonal-in-L "
          "mass matrix discards entirely)")
    coupling_records = []
    for ci in chain_idx:
        params_tf = tf.constant(alm_samples[ci], dtype=tf.float64)
        phi_np = phi_samples[ci].astype(np.float64)

        w_from_stuck, c_stuck_to_healthy, n1 = cross_bin_coupling(
            model, params_tf, phi_np, L_arr, stuck_bin, healthy_bin, args.n_probes, rng
        )
        w_from_healthy, c_healthy_to_stuck, n2 = cross_bin_coupling(
            model, params_tf, phi_np, L_arr, healthy_bin, stuck_bin, args.n_probes, rng
        )
        ratio_stuck = c_stuck_to_healthy / w_from_stuck if w_from_stuck else float("nan")
        ratio_healthy = c_healthy_to_stuck / w_from_healthy if w_from_healthy else float("nan")
        coupling_records.append([w_from_stuck, c_stuck_to_healthy, ratio_stuck,
                                  w_from_healthy, c_healthy_to_stuck, ratio_healthy])
        print(f"  chain point {ci:5d}: "
              f"stuck->healthy coupling ratio={ratio_stuck:.3f} "
              f"(within={w_from_stuck:.3e}, cross={c_stuck_to_healthy:.3e}, n={n1})  |  "
              f"healthy->stuck coupling ratio={ratio_healthy:.3f} "
              f"(within={w_from_healthy:.3e}, cross={c_healthy_to_stuck:.3e}, n={n2})")

    coupling_arr = np.array(coupling_records)
    mean_ratio_stuck = np.nanmean(coupling_arr[:, 2])
    mean_ratio_healthy = np.nanmean(coupling_arr[:, 5])

    print("\n=== Verdict ===")
    print(f"Mean cross/within coupling ratio (stuck bin's perturbations leaking into "
          f"the healthy bin): {mean_ratio_stuck:.3f}")
    print(f"Mean cross/within coupling ratio (healthy bin's perturbations leaking into "
          f"the stuck bin): {mean_ratio_healthy:.3f}")
    if max(mean_ratio_stuck, mean_ratio_healthy) > 0.1:
        print("SIGNIFICANT CROSS-L COUPLING: a diagonal-in-L mass matrix (prior or "
              "Fisher, either one) structurally cannot represent this curvature -- "
              "consistent with every diagonal-preconditioner attempt failing "
              "regardless of tuning. A block or low-rank-correction mass matrix "
              "(not a diagonal one) is the structurally motivated next step, not a "
              "different diagonal estimate or adaptive step size on top of a "
              "diagonal mass matrix.")
    else:
        print("Coupling is small relative to within-bin curvature -- the diagonal "
              "assumption looks locally reasonable here, so the pathology is more "
              "likely a genuine non-Gaussian posterior shape (not fixable by any "
              "quadratic/Euclidean-metric preconditioner) or something outside this "
              "test's scope.")

    np.savez(
        args.out,
        chain_idx=chain_idx,
        bins=np.array(bins),
        diag_records=np.array(diag_records),
        coupling_records=coupling_arr,
        stuck_bin=np.array(stuck_bin),
        healthy_bin=np.array(healthy_bin),
        mean_ratio_stuck=mean_ratio_stuck,
        mean_ratio_healthy=mean_ratio_healthy,
    )
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
