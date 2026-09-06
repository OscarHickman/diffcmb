"""
ROADMAP.md NEXT SESSION (2026-09-02): scoped follow-up to job 11912088's
harvest. Doubling phi_n_lfs (240 -> 480) moved the strict C_L^phiphi SBC
rank from 0.3802 to 0.4196 -- closer to 0.5, but the KS_p got SMALLER
(0.0013 -> 0.00049), and the failure is now concentrated almost entirely in
one bin, [10,30) (KS_p 0.0001; the other three bins individually pass).
More trajectory length is not obviously the next lever for a localized,
ell-dependent residual.

This directly re-runs the cross-L Hessian coupling diagnostic
(diagnose_phi_hessian_coupling.py, 2026-08-17) against the CURRENT failing
bin under the CURRENT production configuration, neither of which the
original script used:

  - Original script hardcoded stuck_bin=(60,lmax), healthy_bin=(10,30) --
    exactly backwards from today's finding, where [10,30) is the one that
    fails the strict SBC rank and the others (including [60,64)) pass.
    That result predates the alm ordering fix, the dof fix, and the proper
    prior, so it is not known to still hold; this script does not assume it
    does.
  - Original script rebuilt a pilot-chain model (pilot_coverage_equilibration
    seed streams). This script rebuilds coverage_ensemble_chain.py's model
    instead (STREAM_TRUTH/STREAM_NOISE/STREAM_CLPP_PRIOR seed streams,
    cl_phiphi_prior_nu on the truth draw), and reads phi_samples/alm_samples
    straight from job 11912088's own chain_r*.npz files -- so the gradient
    probes are evaluated against the EXACT likelihood + prior those samples
    were actually drawn under, not an approximate stand-in.

Method: unchanged from the original script. At a few states along a saved
chain, perturb one packed-phi coordinate in the stuck bin (or the comparison
bin) by eps and take the analytic-gradient finite difference; compare the
response magnitude within the source bin (diagonal-ish) against the response
in the OTHER bin (the coupling a diagonal-in-L mass matrix -- 'prior', the
current production choice -- structurally discards). No new MCMC; each probe
is one lensed-likelihood forward+backward pass at lmax=64 (~seconds).

Verdict guide (same threshold as the original script, ROADMAP.md 2026-08-17):
mean cross/within ratio > 0.1 => significant cross-L coupling, meaning the
current 'prior' (diagonal-in-L) mass matrix cannot represent the curvature
tying [10,30) to whichever bin(s) it's coupled to -- a structural argument
for a non-diagonal preconditioner IF one can be built that does not repeat
'block' (Nystrom)'s failure mode against Block 4 (achievements.md:
'block' + Block 4 ON was tested and lost decisively to 'prior', 0/4 bins vs
2/4 clean + 1 marginal, job 11874976). Small ratio => the pathology is more
likely a genuine non-Gaussian posterior shape in [10,30) specifically, not
fixable by any quadratic/Euclidean-metric preconditioner -- points away from
mass-matrix engineering and toward the likelihood/prior shape itself in that
range.

Usage:
  PYTHONPATH=diffcmb:scripts .venv/bin/python \
      scripts/diagnose_phi_hessian_coupling_l10_30.py \
      --realizations 0,1,2 --n_chain_points 4 --n_probes 6
"""
import argparse
import os

import numpy as np
import tensorflow as tf
from coverage_ensemble_chain import (
    _STREAM_CLPP_PRIOR,
    _STREAM_NOISE,
    _STREAM_TRUTH,
    LCDM_PARAMS,
    _synalm_pair,
    get_cl_phiphi,
)

from diffcmb import CosmologyAdvancedSampling
from diffcmb.alm_utils import packed_sizes
from diffcmb.lensing import _alm_hp_to_packed, lens_map_tf, psi_lensed
from diffcmb.power import call_CAMB_map
from diffcmb.samplers import _alm_index_lm

CHAINDIR = "results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_longtraj"


def rebuild_model_and_data(lmax, nside, noisesig, realization, cl_phiphi_prior_nu):
    """Bit-for-bit reproduction of coverage_ensemble_chain.py's model+data
    construction for one realization, so the probed gradients are exactly
    the ones the saved chain's Block 3 HMC actually sampled against."""
    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=noisesig,
        data_mode="synthetic", dtype=tf.complex128, use_matrixfree_sht=True,
    )
    model._ensure_tf_tensors()

    cl_true = call_CAMB_map(LCDM_PARAMS, lmax)
    cl_phiphi_fid = get_cl_phiphi(lmax)

    nu = float(cl_phiphi_prior_nu)
    rng_prior = np.random.default_rng(_STREAM_CLPP_PRIOR + realization)
    cl_phiphi_true = np.zeros(lmax, dtype=np.float64)
    for ell in range(2, lmax):
        alpha0, beta0 = nu / 2.0, nu * cl_phiphi_fid[ell] / 2.0
        cl_phiphi_true[ell] = beta0 / rng_prior.gamma(alpha0, scale=1.0)

    alm_true_hp, phi_true_hp = _synalm_pair(
        cl_true, cl_phiphi_true, lmax, _STREAM_TRUTH + realization
    )
    alm_true_packed = _alm_hp_to_packed(alm_true_hp, lmax)

    T_lensed_true = lens_map_tf(
        model, tf.constant(alm_true_packed, tf.float64), phi_true_hp
    ).numpy()
    rng_noise = np.random.default_rng(_STREAM_NOISE + realization)
    noisy_map = T_lensed_true + rng_noise.normal(0.0, noisesig, size=model.NPIX)
    model.prior_map = noisy_map
    model.prior_map_masked = tf.convert_to_tensor(
        noisy_map[model.unmasked_idx], dtype=tf.float64
    )
    return model


def analytic_grad(model, params_tf, phi_val):
    phi_var = tf.Variable(phi_val, dtype=tf.float64)
    with tf.GradientTape() as tape:
        val = psi_lensed(model, params_tf, phi_var)
    return tape.gradient(val, phi_var).numpy()


def cross_bin_coupling(model, params_tf, phi_np, L_arr, bin_j, bin_i, n_probes, rng, eps=1e-9):
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
    p.add_argument("--realizations", type=str, default="0,1,2",
                   help="comma-separated realization indices from the "
                        "properprior_longtraj ensemble")
    p.add_argument("--lmax", type=int, default=64)
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--noisesig", type=float, default=1.0)
    p.add_argument("--cl_phiphi_prior_nu", type=float, default=6.0)
    p.add_argument("--n_chain_points", type=int, default=4)
    p.add_argument("--n_probes", type=int, default=6)
    p.add_argument("--out", type=str,
                   default="results/analysis/diagnose_phi_hessian_coupling_l10_30.npz")
    args = p.parse_args()

    stuck_bin = (10, 30)  # fails the strict C_L^phiphi SBC rank, job 11912088
    other_bins = [(2, 10), (30, args.lmax if args.lmax < 60 else 60), (60, args.lmax)]
    other_bins = [(lo, min(hi, args.lmax)) for lo, hi in other_bins if lo < args.lmax]

    n_real = args.lmax * (args.lmax + 1) // 2 - 3
    n_imag = packed_sizes(args.lmax)[1]
    L_arr, _ = _alm_index_lm(args.lmax, n_real, n_imag)

    realizations = [int(x) for x in args.realizations.split(",")]
    rng = np.random.default_rng(12345)

    all_records = {b: [] for b in other_bins}

    for r in realizations:
        chain_path = os.path.join(CHAINDIR, f"chain_r{r:03d}.npz")
        print(f"\n{'='*78}\nRealization {r}: rebuilding model+data, loading {chain_path}")
        model = rebuild_model_and_data(
            args.lmax, args.nside, args.noisesig, r, args.cl_phiphi_prior_nu
        )
        d = np.load(chain_path)
        phi_samples = d["phi_samples"]
        alm_samples = d["alm_samples"]
        n_total = phi_samples.shape[0]
        chain_idx = np.linspace(n_total // 4, n_total - 1, args.n_chain_points, dtype=int)

        for other_bin in other_bins:
            print(f"\n  --- stuck {stuck_bin} <-> {other_bin} ---")
            for ci in chain_idx:
                params_tf = tf.constant(alm_samples[ci], dtype=tf.float64)
                phi_np = phi_samples[ci].astype(np.float64)
                w_s, c_s2o, n1 = cross_bin_coupling(
                    model, params_tf, phi_np, L_arr, stuck_bin, other_bin, args.n_probes, rng
                )
                w_o, c_o2s, n2 = cross_bin_coupling(
                    model, params_tf, phi_np, L_arr, other_bin, stuck_bin, args.n_probes, rng
                )
                ratio_s = c_s2o / w_s if w_s else float("nan")
                ratio_o = c_o2s / w_o if w_o else float("nan")
                all_records[other_bin].append((ratio_s, ratio_o))
                print(f"    chain pt {ci:5d}: stuck->other ratio={ratio_s:.3f} "
                      f"(n={n1})  |  other->stuck ratio={ratio_o:.3f} (n={n2})")

    print(f"\n{'='*78}\n=== Verdict, pooled over {len(realizations)} realizations ===")
    verdict_significant = False
    for other_bin, recs in all_records.items():
        arr = np.array(recs)
        mean_s2o = np.nanmean(arr[:, 0])
        mean_o2s = np.nanmean(arr[:, 1])
        flag = " <-- SIGNIFICANT" if max(mean_s2o, mean_o2s) > 0.1 else ""
        print(f"  {stuck_bin} <-> {other_bin}: stuck->other={mean_s2o:.3f}  "
              f"other->stuck={mean_o2s:.3f}{flag}")
        if max(mean_s2o, mean_o2s) > 0.1:
            verdict_significant = True

    print()
    if verdict_significant:
        print("SIGNIFICANT CROSS-L COUPLING involving [10,30): the current "
              "'prior' (diagonal-in-L) mass matrix cannot represent this "
              "curvature. Consistent with a structural, not purely-mixing, "
              "explanation for the localized SBC failure. NOTE: 'block' "
              "(Nystrom cross-L correction) was already tested with Block 4 "
              "ON and lost decisively to 'prior' (achievements.md, job "
              "11874976, 0/4 bins pass) -- so this does not by itself "
              "recommend reviving 'block' as-is; it motivates checking WHY "
              "block failed (the untested rank-deficiency hypothesis in "
              "achievements.md) before trying any non-diagonal preconditioner "
              "again.")
    else:
        print("Coupling involving [10,30) is small relative to within-bin "
              "curvature. The diagonal-in-L assumption looks locally "
              "reasonable, so the localized SBC failure is more likely a "
              "genuine non-Gaussian posterior shape in that bin (not fixable "
              "by any quadratic/Euclidean-metric preconditioner) or an "
              "artifact of the proper prior's own shape at those L -- look "
              "at Block 4's prior construction and cl_phiphi_true's draw in "
              "[10,30) specifically, not at Block 3's geometry.")

    np.savez(
        args.out,
        stuck_bin=np.array(stuck_bin),
        other_bins=np.array(other_bins),
        realizations=np.array(realizations),
        **{f"records_{lo}_{hi}": np.array(all_records[(lo, hi)])
           for (lo, hi) in other_bins},
    )
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
