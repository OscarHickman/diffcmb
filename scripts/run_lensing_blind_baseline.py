"""
Commander-style lensing-blind Gibbs baseline for the C_ℓ^TT bias-reduction figure.

ROADMAP.md §2: "C_ℓ^TT bias reduction vs a lensing-blind (Commander-style) analysis
of the same sims."

This script runs the Phase 0 unlensed Gibbs sampler (blocks 1+2 only: exact
C_ℓ|alm inverse-Gamma draw + HMC alm|C_ℓ) on the IDENTICAL simulation that
the MCLMC equilibration pilot uses — same seed, lmax, nside, noise level —
so the recovered C_ℓ^TT posterior can be directly compared against the joint
lensing-aware result once that chain is equilibrated.

"Lensing-blind" here means the sampler uses the *unlensed* (Phase 0) posterior:
it knows about no φ, treats the lensed map as if it were an unlensed CMB
realization, and fits the best-fitting power spectrum to the lensed data. This
is exactly the Commander paradigm: map + C_ℓ Gibbs, no lensing model.

WHY NOW: the comparison figure (lensing-aware C_ℓ^TT vs lensing-blind C_ℓ^TT)
is independent of the equilibration gate. We can generate the reference chain
now; the joint posterior side will slot in once the phi-block equilibration is
resolved.

DESIGN:
- Same simulation generation as pilot_coverage_equilibration.py (seed=0,
  lmax=128, nside=128, noisesig=1.0, LCDM_PARAMS). The simulation is generated
  from scratch here (not loaded from a checkpoint) to keep this script
  self-contained, but the same random seed ensures the sky + noise realization
  are bit-for-bit identical.
- MAP warm-start for alm (same map_steps/map_lr as the pilot) — clean start.
- No phi block (cl_phiphi_full=None). No Block 4. Pure Phase 0.
- Long chain: n_burnin=500, n_samples=3000 — more than enough to characterize
  the posterior C_ℓ^TT well; the lensing bias is in the mean, not the variance.
- Saves the full alm_samples and C_ℓ posteriors in the same format as the
  pilot chain for easy comparison.

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/run_lensing_blind_baseline.py \\
      --lmax 128 --nside 128 --seed 0 \\
      --out results/analysis/lensing_blind_baseline_lmax128.npz
"""

import argparse
import time

import healpy as hp
import numpy as np
import tensorflow as tf

from diffcmb import CosmologyAdvancedSampling, run_gibbs_chain
from diffcmb.lensing import _alm_hp_to_packed, lens_map_tf
from diffcmb.power import call_CAMB_map, fiducial_spectra
from diffcmb.samplers import find_map_estimate

# Identical to pilot_coverage_equilibration.py
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lmax", type=int, default=128)
    p.add_argument("--nside", type=int, default=128)
    p.add_argument("--noisesig", type=float, default=1.0)
    p.add_argument("--lensing_operator", choices=("exact", "bilinear"), default="exact",
                   help="Forward lensing operator. 'exact' (default since 2026-09-23) "
                        "evaluates the alm at the deflected positions and lenses by "
                        "+grad(phi); 'bilinear' is the legacy interpolation (and "
                        "-grad(phi) sign) that every earlier chain used. ROADMAP D3.")
    p.add_argument("--fiducial", choices=("corrected", "legacy"), default="corrected",
                   help="'corrected' = power.fiducial_spectra (physical densities, "
                        "raw C_l, unlensed TT); 'legacy' = the pre-2026-09-23 "
                        "call_CAMB_map spectra, for reproducing old chains only.")
    p.add_argument("--n_burnin", type=int, default=500)
    p.add_argument("--n_samples", type=int, default=3000)
    p.add_argument("--map_steps", type=int, default=2000)
    p.add_argument("--map_lr", type=float, default=0.01)
    p.add_argument("--hmc_step_size", type=float, default=0.01)
    p.add_argument("--n_lfs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0,
                   help="Must match the pilot chain seed to get the identical simulation")
    p.add_argument("--checkpoint_path", type=str,
                   default="results/analysis/lensing_blind_baseline_lmax128_ckpt.npz")
    p.add_argument("--checkpoint_every", type=int, default=100)
    p.add_argument("--out", type=str,
                   default="results/analysis/lensing_blind_baseline_lmax128.npz")
    args = p.parse_args()

    lmax, nside = args.lmax, args.nside

    print(f"=== Commander-style lensing-blind baseline: lmax={lmax}, nside={nside} ===")
    print("Phase 0 unlensed Gibbs (alm + C_ℓ only; no phi block).")
    print("Simulation generated with the same seed as the MCLMC pilot (seed=0).\n")

    print("Building matrix-free-SHT model...")
    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=args.noisesig,
        data_mode="synthetic", dtype=tf.complex128, use_matrixfree_sht=True,
        lensing_operator=args.lensing_operator,
    )
    model._ensure_tf_tensors()

    print("Drawing (alm_true, phi_true) from CAMB spectra (same seed as pilot)...")
    if args.fiducial == "corrected":
        cl_true, cl_phiphi_true = fiducial_spectra(lmax)
    else:
        cl_true = call_CAMB_map(LCDM_PARAMS, lmax)
        cl_phiphi_true = get_cl_phiphi(lmax)

    # Identical RNG setup as pilot_coverage_equilibration.py (seed=0):
    # rng_truth seeded from args.seed, rng_noise from args.seed + 20_000
    rng_truth = np.random.default_rng(args.seed)
    rng_noise = np.random.default_rng(args.seed + 20_000)

    np.random.seed(rng_truth.integers(0, 2**31 - 1))
    alm_true_hp = hp.synalm(cl_true, lmax=lmax - 1, new=True).astype(np.complex128)
    phi_true_hp = hp.synalm(cl_phiphi_true, lmax=lmax - 1, new=True).astype(np.complex128)
    alm_true_packed = _alm_hp_to_packed(alm_true_hp, lmax)

    print("Lensing the true sky and adding noise (replicates the pilot simulation)...")
    T_lensed = lens_map_tf(
        model, tf.constant(alm_true_packed, tf.float64), phi_true_hp
    ).numpy()
    noisy_map = T_lensed + rng_noise.normal(0.0, args.noisesig, size=model.NPIX)
    model.prior_map = noisy_map
    model.prior_map_masked = tf.convert_to_tensor(
        noisy_map[model.unmasked_idx], dtype=tf.float64
    )

    # MAP warm-start (data-driven, never sees truth — same as pilot)
    print(f"Finding MAP estimate ({args.map_steps} Adam steps, lr={args.map_lr})...")
    t_map = time.time()
    x0 = np.asarray(
        find_map_estimate(model, n_steps=args.map_steps, learning_rate=args.map_lr),
        dtype=np.float64,
    )
    print(f"  MAP done in {time.time() - t_map:.1f}s")

    print(f"\nRunning lensing-blind Gibbs: n_burnin={args.n_burnin}, "
          f"n_samples={args.n_samples}, NO phi block...")
    t0 = time.time()
    # No cl_phiphi_full → no phi block → pure Phase 0 Commander-style sampler
    result = run_gibbs_chain(
        model,
        n_samples=args.n_samples,
        n_burnin=args.n_burnin,
        hmc_step_size=args.hmc_step_size,
        n_lfs=args.n_lfs,
        initial_params=x0,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
        checkpoint_every=args.checkpoint_every,
    )
    # run_gibbs_chain with no phi block (cl_phiphi_full=None, sample_phi False)
    # returns a 4-tuple: (samples, logp, accepts, final_step) -- no phi_samples slot.
    samples, logp, accepts, final_step = result[:4]
    elapsed = time.time() - t0
    n_collected = max(1, len(samples))

    print(f"\n  Done in {elapsed / 3600:.2f}h  ({elapsed / n_collected:.1f}s/sweep)")
    print(f"  alm-block accept rate: {accepts.mean():.3f}")

    # Reconstruct C_ℓ posterior from alm_samples
    # samples has shape (n_samples, n_lncl + n_alm); first lmax-2 elements are ln(C_ℓ)
    n_lncl = lmax - 2
    lncl_samples = samples[:, :n_lncl]          # ln(C_ℓ), ℓ=2..lmax-1
    cl_samples = np.exp(lncl_samples)            # C_ℓ per sample

    cl_mean = cl_samples.mean(axis=0)
    cl_std = cl_samples.std(axis=0)
    ell_arr = np.arange(2, lmax)

    print("\n  C_ℓ^TT posterior (ℓ × (ℓ+1) / 2π units, sample mean ± std):")
    for i in range(0, min(10, len(ell_arr))):
        ell = ell_arr[i]
        norm = ell * (ell + 1) / (2 * np.pi)
        cl_true_here = cl_true[ell] if ell < len(cl_true) else np.nan
        print(f"    ℓ={ell:4d}  D_ℓ^blind={norm * cl_mean[i]:.4e} ± {norm * cl_std[i]:.4e}"
              f"  D_ℓ^true={norm * cl_true_here:.4e}")

    np.savez(
        args.out,
        alm_samples=samples,
        cl_samples=cl_samples,
        lncl_samples=lncl_samples,
        logp=logp,
        accepts=accepts,
        cl_true=cl_true,
        cl_phiphi_true=cl_phiphi_true,
        alm_true_packed=alm_true_packed,
        ell_arr=ell_arr,
        seconds_total=elapsed,
        seconds_per_sweep=elapsed / n_collected,
        lmax=lmax,
        nside=nside,
        seed=args.seed,
        lensing_operator=args.lensing_operator, fiducial=args.fiducial,
        noisesig=args.noisesig,
    )
    print(f"\nSaved lensing-blind baseline chain to {args.out}")
    print("Compare cl_samples[:,ℓ-2] against the joint-lensing-aware C_ℓ posterior")
    print("for the C_ℓ^TT bias-reduction figure (ROADMAP.md §2.3).")


if __name__ == "__main__":
    main()
