"""
ROADMAP.md Section 1, "Next up" item 1: simulation validation of the
matrix-free 3-block Gibbs sampler (C_l | alm exact; alm | C_l, phi HMC;
phi | alm, C_l HMC) at lmax=300, now that the Block 3 lensing port has
passed its production-scale smoke test (scripts/smoke_lmax300_matrixfree_lensing.py,
job 11611338, 2026-07-18).

Draws a synthetic full-sky lensed sky with known (alm_true, phi_true) at the
model's fixed LCDM cosmology, runs a single chain warm-started at the truth
(as in gates 1/2 -- this isolates the conditional's recovery/mixing from
burn-in-to-mode time, which is a separate concern already addressed by
find_map_estimate in production chains), and reports:

  1. Point agreement: posterior-mean C_l^TT and C_L^phiphi vs the true
     input spectra, and posterior-mean phi_lm vs phi_true_lm, binned by l.
  2. A per-bin z-score ((posterior mean - truth) / posterior std) as a
     single-realization coverage proxy -- not a full rank/coverage test
     over many independent simulations (that requires O(10) independent
     chains at this wall-clock cost and is future work), but a necessary
     condition: z-scores should scatter roughly like a standard normal, not
     show a systematic multi-sigma bias at any l.

Cost note: at n_lfs=10 for both HMC blocks, ~29s/sweep (tuning job 11612586,
lmax=300/nside=256) -- much cheaper than the n_lfs=20 smoke-run's 54s/sweep
with only a modest expected mixing cost, since gate 1 found the joint
(alm,phi) mixing MARGINAL rather than pathological at this configuration's
smaller-scale analogue. Checkpointed so a 24h SLURM walltime limit can be
resumed across multiple submissions.

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/validate_sim_lmax300_lensing.py \
    --lmax 300 --nside 256 --n_burnin 200 --n_samples 1000 --n_lfs 10 --phi_n_lfs 10 \
    --checkpoint_path results/analysis/validate_sim_lmax300_ckpt.npz
"""
import argparse
import time

import healpy as hp
import numpy as np
import tensorflow as tf

from diffcmb import CosmologyAdvancedSampling, run_gibbs_chain
from diffcmb.alm_utils import packed_sizes
from diffcmb.lensing import _alm_hp_to_packed, lens_map_tf
from diffcmb.power import call_CAMB_map
from diffcmb.samplers import _alm_index_lm

LCDM_PARAMS = [67.74, 0.0486, 0.2589, 0.06, 0.0, 0.066]


def get_cl_phiphi(lmax):
    import camb

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=LCDM_PARAMS[0], ombh2=LCDM_PARAMS[1], omch2=LCDM_PARAMS[2],
        mnu=LCDM_PARAMS[3], omk=LCDM_PARAMS[4], tau=LCDM_PARAMS[5],
    )
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    results = camb.get_results(pars)
    dl_pp = results.get_lens_potential_cls(lmax=lmax - 1)[:, 0]
    cl_pp = np.zeros(lmax, dtype=np.float64)
    for ell in range(2, lmax):
        if ell < len(dl_pp):
            norm = (ell * (ell + 1)) ** 2 / (2.0 * np.pi)
            cl_pp[ell] = dl_pp[ell] / norm
    return cl_pp


def binned_power_recovery(samples, true_packed, L_arr, ell_bins, label):
    """For each l-bin, compare posterior-mean power to the truth's power and
    report a z-score using the posterior std as the uncertainty."""
    print(f"\n--- {label}: binned power recovery ---")
    rows = []
    power_samples = samples ** 2  # (n_samples, n_coeff), per-coefficient power
    true_power = true_packed ** 2
    for lo, hi in ell_bins:
        mask = (L_arr >= lo) & (L_arr < hi)
        if not mask.any():
            continue
        post_power_mean = power_samples[:, mask].mean(axis=1)  # per-sweep bin-avg power
        post_mean = post_power_mean.mean()
        post_std = post_power_mean.std()
        truth_mean = true_power[mask].mean()
        z = (post_mean - truth_mean) / post_std if post_std > 0 else np.nan
        frac_bias = 100.0 * (post_mean - truth_mean) / truth_mean if truth_mean > 0 else np.nan
        print(f"  l=[{lo:4d},{hi:4d})  truth={truth_mean:.4e}  "
              f"post_mean={post_mean:.4e}  post_std={post_std:.4e}  "
              f"z={z:6.2f}  frac_bias={frac_bias:7.2f}%")
        rows.append((lo, hi, truth_mean, post_mean, post_std, z, frac_bias))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lmax", type=int, default=300)
    p.add_argument("--nside", type=int, default=256)
    p.add_argument("--noisesig", type=float, default=1.0)
    p.add_argument("--n_burnin", type=int, default=200)
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--hmc_step_size", type=float, default=0.01)
    p.add_argument("--n_lfs", type=int, default=10)
    p.add_argument("--phi_hmc_step_size", type=float, default=0.001)
    p.add_argument("--phi_n_lfs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint_path", type=str,
                    default="results/analysis/validate_sim_lmax300_ckpt.npz")
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument("--out", type=str,
                    default="results/analysis/validate_sim_lmax300.npz")
    args = p.parse_args()

    lmax, nside = args.lmax, args.nside
    rng = np.random.default_rng(args.seed)

    print(f"=== lmax={lmax} simulation validation: matrix-free 3-block Gibbs "
          f"(nside={nside}) ===\n")

    print("Building matrix-free-SHT model (synthetic, full-sky)...")
    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=args.noisesig,
        data_mode="synthetic", dtype=tf.complex128, use_matrixfree_sht=True,
    )
    model._ensure_tf_tensors()
    assert len(model.unmasked_idx) == model.NPIX, (
        "validation assumes full-sky (synthetic) data -- masked-sky matrix-free "
        "lensing is not yet validated (ROADMAP.md Section 1)"
    )

    print("Drawing (alm_true, phi_true) from CAMB spectra at the fixed LCDM cosmology...")
    cl_true = call_CAMB_map(LCDM_PARAMS, lmax)
    cl_phiphi_true = get_cl_phiphi(lmax)

    alm_true_hp = hp.synalm(cl_true, lmax=lmax - 1, new=True).astype(np.complex128)
    phi_true_hp = hp.synalm(cl_phiphi_true, lmax=lmax - 1, new=True).astype(np.complex128)
    alm_true_packed = _alm_hp_to_packed(alm_true_hp, lmax)
    phi_true_packed = _alm_hp_to_packed(phi_true_hp, lmax)

    print("Lensing the true sky through the matrix-free forward operator and adding noise...")
    T_lensed_true = lens_map_tf(
        model, tf.constant(alm_true_packed, tf.float64), phi_true_hp
    ).numpy()
    if not np.all(np.isfinite(T_lensed_true)):
        raise RuntimeError("lens_map_tf produced NaN/Inf -- aborting validation run")

    noisy_map = T_lensed_true + rng.normal(0.0, args.noisesig, size=model.NPIX)
    model.prior_map = noisy_map
    model.prior_map_masked = tf.convert_to_tensor(
        noisy_map[model.unmasked_idx], dtype=tf.float64
    )

    x0 = np.concatenate([np.log(cl_true[2:lmax]), alm_true_packed])

    print(f"\nRunning 3-block Gibbs chain: n_burnin={args.n_burnin}, "
          f"n_samples={args.n_samples}, n_lfs={args.n_lfs}, "
          f"phi_n_lfs={args.phi_n_lfs}, warm-started at truth "
          f"(checkpoint={args.checkpoint_path})...")
    t0 = time.time()
    samples, phi_samples, logp, accepts, final_step = run_gibbs_chain(
        model,
        n_samples=args.n_samples,
        n_burnin=args.n_burnin,
        hmc_step_size=args.hmc_step_size,
        n_lfs=args.n_lfs,
        initial_params=x0,
        cl_phiphi_full=cl_phiphi_true,
        phi_initial=phi_true_packed,
        phi_hmc_step_size=args.phi_hmc_step_size,
        phi_n_lfs=args.phi_n_lfs,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
        checkpoint_every=args.checkpoint_every,
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed / 3600:.2f}h ({elapsed / max(1, len(samples)):.2f}s/collected-sample)")
    print(f"  alm-block accept rate: {accepts.mean():.3f}")

    if not (np.all(np.isfinite(samples)) and np.all(np.isfinite(phi_samples))
            and np.all(np.isfinite(logp))):
        raise RuntimeError("NaN/Inf in samples/phi_samples/logp -- validation run FAILED")

    lmax_v = lmax
    n_lncl = lmax_v - 2
    n_real = lmax_v * (lmax_v + 1) // 2 - 3
    n_imag = packed_sizes(lmax_v)[1]
    L_arr, _m_arr = _alm_index_lm(lmax_v, n_real, n_imag)

    alm_part = samples[:, n_lncl:]
    lncl_part = samples[:, :n_lncl]
    cl_post = np.exp(lncl_part)  # (n_samples, lmax-2), l=2..lmax-1

    ell_bins = [(2, 10), (10, 30), (30, 60), (60, 100), (100, 150),
                (150, 200), (200, 250), (250, 300)]

    # Truth for this check is the *realized* power in alm_true_packed (what the
    # exact inverse-Gamma C_l|alm conditional actually targets: C_l|alm ~
    # InvGamma(l-0.5, S_l/2), model.py::sample_cl_given_alm), not the CAMB
    # ensemble-average cl_true -- at low l (few (2l+1) modes) cosmic variance
    # makes those two disagree by O(1), which would masquerade as a bias here.
    # model.compute_sl_np gives the exact S_l = sum_m |a_lm|^2 with the correct
    # m=0-vs-m>0 packed-real/imag weighting (see hpalmtocl's convention).
    S_l = model.compute_sl_np(alm_true_packed)
    realized_cl = np.zeros(lmax_v, dtype=np.float64)
    for ell in range(2, lmax_v):
        realized_cl[ell] = S_l[ell] / (2 * ell + 1)

    print("\n=== C_l^TT recovery (posterior draws vs the realized power in "
          "alm_true, not the CAMB ensemble spectrum -- see cosmic-variance note above) ===")
    cl_rows = []
    for lo, hi in ell_bins:
        ells = np.arange(max(2, lo), min(lmax_v, hi))
        if len(ells) == 0:
            continue
        idx = ells - 2
        post_mean = cl_post[:, idx].mean()
        post_std = cl_post[:, idx].mean(axis=1).std()
        truth_mean = realized_cl[ells].mean()
        z = (post_mean - truth_mean) / post_std if post_std > 0 else np.nan
        frac_bias = 100.0 * (post_mean - truth_mean) / truth_mean if truth_mean > 0 else np.nan
        print(f"  l=[{lo:4d},{hi:4d})  truth={truth_mean:.4e}  "
              f"post_mean={post_mean:.4e}  z={z:6.2f}  frac_bias={frac_bias:7.2f}%")
        cl_rows.append((lo, hi, truth_mean, post_mean, post_std, z, frac_bias))

    alm_rows = binned_power_recovery(alm_part, alm_true_packed, L_arr, ell_bins,
                                      "alm block (unlensed T sky)")
    phi_rows = binned_power_recovery(phi_samples, phi_true_packed, L_arr, ell_bins,
                                      "phi block (lensing potential)")

    print("\n=== Verdict ===")
    all_z = np.array([r[5] for r in cl_rows + alm_rows + phi_rows])
    all_z = all_z[np.isfinite(all_z)]
    worst_z = np.abs(all_z).max() if len(all_z) else np.nan
    if not len(all_z):
        print("INCONCLUSIVE: no finite z-scores computed (posterior std collapsed to zero "
              "somewhere) -- inspect the raw per-bin output above.")
    elif worst_z < 3.0:
        print(f"PASS (point-agreement + single-realization coverage proxy): worst "
              f"|z|={worst_z:.2f} across all C_l^TT/alm/phi bins -- no evidence of "
              f"systematic bias beyond posterior uncertainty. NOTE: this is a single "
              f"realization, not a full rank/coverage test over many independent sims "
              f"(ROADMAP.md Section 1 still calls for that before the paper).")
    else:
        print(f"FAIL/INVESTIGATE: worst |z|={worst_z:.2f} exceeds 3 sigma -- either a "
              f"real bias (re-check the lensing/likelihood implementation) or the chain "
              f"hasn't mixed enough for the posterior std estimate to be trustworthy "
              f"(re-check ESS/accept rates above before concluding bias).")

    np.savez(
        args.out,
        alm_samples=samples, phi_samples=phi_samples, logp=logp, accepts=accepts,
        cl_true=cl_true, cl_phiphi_true=cl_phiphi_true,
        alm_true_packed=alm_true_packed, phi_true_packed=phi_true_packed,
        cl_rows=np.array(cl_rows, dtype=object),
        alm_rows=np.array(alm_rows, dtype=object),
        phi_rows=np.array(phi_rows, dtype=object),
        seconds_total=elapsed,
    )
    print(f"\nSaved chain + truth + recovery stats to {args.out}")


if __name__ == "__main__":
    main()
