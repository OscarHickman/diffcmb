"""
ROADMAP.md Section 1, Phase 2 pre-production gate 2: Block-2 exactness
experiment at small lmax.

Weak gravitational lensing is close to norm-preserving (it remaps power
across scales but barely changes the total unlensed-vs-lensed likelihood
curvature at the S/N reachable with a lmax<=50, single-map synthetic test).
This raises the possibility that Block 2 (alm | C_l, phi) can be replaced by
`sample_alm_cg`'s *exact* Gaussian draw -- calibrated against the unlensed
forward operator (`model._psi_tf_raw`, no phi dependence) -- while still
producing an unbiased alm/C_l posterior once phi is genuinely present in the
data. If so, HMC is only needed for Block 3 (phi | alm, C_l): cleaner and
much faster than running HMC on both alm and phi (gate 1's configuration).

This is deliberately testing an approximation for bias, not assuming it is
correct: samplers.py's alm_sampler='cg' + cl_phiphi_full combination (added
for this gate) draws exact-Gaussian alm ignoring lensing, then draws phi via
HMC against the correct lensed likelihood (log_prob_phi_block) -- see the
comment at samplers.py's run_gibbs_chain guard.

Method
------
1. Reuse gate 1's synthetic-sky setup: draw (alm_true, phi_true) from CAMB
   spectra, lens through the model's own forward operator, add noise.
2. Run the 3-block Gibbs chain with alm_sampler='cg' (Block 2 exact, ignores
   lensing) + phi HMC (Block 3, correct lensing), warm-started at the truth.
3. Compare the resulting C_l^TT posterior mean against the true input
   spectrum (fractional bias per multipole) and against gate 1's all-HMC
   (correct-likelihood) reference chain if available, as the direct
   contrast on whether ignoring lensing in Block 2 introduces bias.

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/archive/gate_block2_exactness_check.py \
    --lmax 50 --nside 64 --n_burnin 500 --n_samples 3000
"""
import argparse
import time

import healpy as hp
import numpy as np
import tensorflow as tf

from diffcmb import CosmologyAdvancedSampling, run_gibbs_chain
from diffcmb.lensing import _alm_hp_to_packed, lens_map_tf
from diffcmb.power import call_CAMB_map

LCDM_PARAMS = [67.74, 0.0486, 0.2589, 0.06, 0.0, 0.066]


def get_cl_phiphi(lmax):
    """CAMB lensing-potential power spectrum C_L^phiphi (same convention as
    gate_phi_alm_mixing_check.py's get_cl_phiphi)."""
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
    p.add_argument("--lmax", type=int, default=50)
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--noisesig", type=float, default=1.0)
    p.add_argument("--n_burnin", type=int, default=500)
    p.add_argument("--n_samples", type=int, default=3000)
    p.add_argument("--n_pcg_iter", type=int, default=50)
    p.add_argument("--phi_hmc_step_size", type=float, default=0.05)
    p.add_argument("--phi_n_lfs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/analysis/block2_exactness_check.npz")
    args = p.parse_args()

    lmax, nside = args.lmax, args.nside
    rng = np.random.default_rng(args.seed)

    print(f"=== Phase 2 pre-production gate 2: Block-2 exactness check "
          f"(lmax={lmax}, nside={nside}) ===\n")

    print("Building model (dense SHT -- Block 3 requires model.sph_parts)...")
    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=args.noisesig,
        data_mode="synthetic", dtype=tf.complex128,
    )
    model._ensure_tf_tensors()
    assert len(model.sph_parts) == 1, (
        "gate script assumes a single sph_parts chunk at this lmax/nside scale"
    )

    print("Drawing (alm_true, phi_true) from CAMB spectra at the fixed LCDM cosmology...")
    cl_true = call_CAMB_map(LCDM_PARAMS, lmax)
    cl_phiphi_true = get_cl_phiphi(lmax)

    alm_true_hp = hp.synalm(cl_true, lmax=lmax - 1, new=True).astype(np.complex128)
    phi_true_hp = hp.synalm(cl_phiphi_true, lmax=lmax - 1, new=True).astype(np.complex128)
    alm_true_packed = _alm_hp_to_packed(alm_true_hp, lmax)
    phi_true_packed = _alm_hp_to_packed(phi_true_hp, lmax)

    print("Lensing the true sky through the model's own forward operator and adding noise...")
    T_lensed_true = lens_map_tf(
        model, tf.constant(alm_true_packed, tf.float64), phi_true_hp
    ).numpy()
    noisy_map = T_lensed_true + rng.normal(0.0, args.noisesig, size=model.NPIX)
    model.prior_map = noisy_map
    model.prior_map_parts = [tf.convert_to_tensor(noisy_map[model.unmasked_idx], dtype=tf.float64)]

    n_lncl = lmax - 2
    x0 = np.concatenate([np.log(cl_true[2:lmax]), alm_true_packed])

    print(f"\nRunning 3-block Gibbs chain with alm_sampler='cg' (Block 2 ignores "
          f"lensing) + phi HMC (Block 3, correct lensing): n_burnin={args.n_burnin}, "
          f"n_samples={args.n_samples}, warm-started at truth...")
    t0 = time.time()
    samples, phi_samples, logp, accepts, final_step = run_gibbs_chain(
        model,
        n_samples=args.n_samples,
        n_burnin=args.n_burnin,
        alm_sampler="cg",
        n_pcg_iter=args.n_pcg_iter,
        initial_params=x0,
        cl_phiphi_full=cl_phiphi_true,
        phi_initial=phi_true_packed,
        phi_hmc_step_size=args.phi_hmc_step_size,
        phi_n_lfs=args.phi_n_lfs,
        seed=args.seed,
    )
    print(f"  done in {(time.time()-t0)/60:.1f} min")
    print(f"  phi-block accept rate: {accepts.mean():.3f} (n/a for Block 2 -- 'cg' has no accept/reject)")

    cl_post = np.exp(samples[:, :n_lncl])
    cl_post_mean = cl_post.mean(axis=0)
    frac_bias = cl_post_mean / cl_true[2:lmax] - 1.0

    print("\n--- C_l^TT posterior mean vs truth (Block 2 ignoring lensing) ---")
    probe_ells = sorted({max(2, min(lmax - 1, v)) for v in
                          np.linspace(2, lmax - 1, 8).round().astype(int)})
    for ell in probe_ells:
        idx = ell - 2
        print(f"  l={ell:4d}  C_l_true={cl_true[ell]:12.4e}  "
              f"C_l_post_mean={cl_post_mean[idx]:12.4e}  frac_bias={frac_bias[idx]:+7.3%}")

    worst_bias = np.abs(frac_bias).max()
    print("\n=== Verdict ===")
    print(f"Worst |fractional bias| in C_l^TT across l=2..{lmax - 1}: {worst_bias:.3%}")
    if worst_bias < 0.05:
        print("PASS: exact unlensed Block-2 draw shows no material bias at this lmax/S-N -- "
              "HMC can confine to Block 3 (phi) only. Cleaner and faster; re-validate at "
              "the intended production lmax before relying on this in Phase 2 chains.")
    elif worst_bias < 0.15:
        print("MARGINAL: some bias from ignoring lensing in Block 2 -- usable only if this "
              "level of C_l^TT bias is acceptable for the target science case; otherwise keep "
              "HMC on both blocks (gate 1's configuration).")
    else:
        print("FAIL: ignoring lensing in Block 2 introduces material C_l^TT bias -- keep HMC "
              "on both alm and phi blocks (gate 1's configuration); this shortcut does not "
              "survive lensing at this S/N.")

    np.savez(
        args.out,
        alm_samples=samples, phi_samples=phi_samples, logp=logp, accepts=accepts,
        cl_true=cl_true, cl_phiphi_true=cl_phiphi_true,
        alm_true_packed=alm_true_packed, phi_true_packed=phi_true_packed,
        cl_post_mean=cl_post_mean, frac_bias=frac_bias, probe_ells=probe_ells,
    )
    print(f"\nSaved chain + truth to {args.out}")


if __name__ == "__main__":
    main()
