"""
ROADMAP.md Section 1, "Next up" item 1: lmax=300 smoke run of the matrix-free
Block 3 lensing port.

The port (lensing.py::lens_map_tf/psi_lensed branching on
model.use_matrixfree_sht, using sht_ducc.py::full_synthesis_tf) was validated
at lmax=50 (tests/test_lensing.py, gate 1/2 scripts) and against the dense
reference to numerical precision, but every test so far ran at small lmax.
This script is *not* a new mixing/exactness gate -- it is a cheap
production-scale smoke test: does the matrix-free 3-block Gibbs chain
(C_l | alm exact; alm | C_l, phi HMC; phi | alm, C_l HMC) run at lmax=300
without NaNs/Infs, in reasonable per-sweep wall-clock time, with sane accept
rates? Mirrors gate_phi_alm_mixing_check.py's structure (synthetic full-sky
data, warm-started at truth) but swaps the dense model for
use_matrixfree_sht=True and keeps the chain short (this is a smoke test, not
a mixing/convergence claim -- that's Section 1's next deliverable).

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/archive/smoke_lmax300_matrixfree_lensing.py \
    --lmax 300 --nside 256 --n_burnin 20 --n_samples 50
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
    """CAMB lensing-potential power spectrum C_L^phiphi (see
    gate_phi_alm_mixing_check.py for the same helper)."""
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
    p.add_argument("--lmax", type=int, default=300)
    p.add_argument("--nside", type=int, default=256)
    p.add_argument("--noisesig", type=float, default=1.0)
    p.add_argument("--n_burnin", type=int, default=20)
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--hmc_step_size", type=float, default=0.01)
    p.add_argument("--n_lfs", type=int, default=20)
    p.add_argument("--phi_hmc_step_size", type=float, default=0.01)
    p.add_argument("--phi_n_lfs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str,
                    default="results/analysis/smoke_lmax300_matrixfree_lensing.npz")
    args = p.parse_args()

    lmax, nside = args.lmax, args.nside

    print(f"=== lmax={lmax} smoke run: matrix-free Block 3 lensing "
          f"(nside={nside}) ===\n")

    print("Building matrix-free-SHT model (synthetic, full-sky)...")
    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=args.noisesig,
        data_mode="synthetic", dtype=tf.complex128, use_matrixfree_sht=True,
    )
    model._ensure_tf_tensors()
    assert len(model.unmasked_idx) == model.NPIX, (
        "smoke run assumes full-sky (synthetic) data -- masked-sky matrix-free "
        "lensing is not yet validated (ROADMAP.md Section 1)"
    )

    print("Drawing (alm_true, phi_true) from CAMB spectra at the fixed LCDM cosmology...")
    rng = np.random.default_rng(args.seed)
    cl_true = call_CAMB_map(LCDM_PARAMS, lmax)
    cl_phiphi_true = get_cl_phiphi(lmax)

    alm_true_hp = hp.synalm(cl_true, lmax=lmax - 1, new=True).astype(np.complex128)
    phi_true_hp = hp.synalm(cl_phiphi_true, lmax=lmax - 1, new=True).astype(np.complex128)
    alm_true_packed = _alm_hp_to_packed(alm_true_hp, lmax)
    phi_true_packed = _alm_hp_to_packed(phi_true_hp, lmax)

    print("Lensing the true sky through the matrix-free forward operator "
          "and adding noise...")
    t_sht0 = time.time()
    T_lensed_true = lens_map_tf(
        model, tf.constant(alm_true_packed, tf.float64), phi_true_hp
    ).numpy()
    print(f"  single lens_map_tf forward pass: {time.time() - t_sht0:.2f}s")

    if not np.all(np.isfinite(T_lensed_true)):
        raise RuntimeError("lens_map_tf produced NaN/Inf at lmax=300 -- smoke test FAILS")

    noisy_map = T_lensed_true + rng.normal(0.0, args.noisesig, size=model.NPIX)
    model.prior_map = noisy_map
    model.prior_map_masked = tf.convert_to_tensor(
        noisy_map[model.unmasked_idx], dtype=tf.float64
    )

    x0 = np.concatenate([np.log(cl_true[2:lmax]), alm_true_packed])

    print(f"\nRunning short 3-block Gibbs chain: n_burnin={args.n_burnin}, "
          f"n_samples={args.n_samples}, warm-started at truth...")
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
    )
    elapsed = time.time() - t0
    n_sweeps = args.n_burnin + args.n_samples
    print(f"  done in {elapsed / 60:.1f} min ({elapsed / n_sweeps:.2f}s/sweep, "
          f"{n_sweeps} sweeps)")
    print(f"  alm-block accept rate: {accepts.mean():.3f}")

    print("\n=== Verdict ===")
    finite_ok = (
        np.all(np.isfinite(samples)) and np.all(np.isfinite(phi_samples))
        and np.all(np.isfinite(logp))
    )
    accept_ok = accepts.mean() > 0.02
    if not finite_ok:
        print("FAIL: NaN/Inf encountered in samples, phi_samples, or logp at lmax=300.")
    elif not accept_ok:
        print(f"FAIL: alm-block accept rate {accepts.mean():.3f} near zero -- "
              "step size mistuned for this scale, not a correctness failure but "
              "blocks trusting downstream numbers.")
    else:
        print("PASS: matrix-free Block 3 lensing runs cleanly at lmax=300 -- "
              "finite samples/logp throughout, sane accept rate, "
              f"{elapsed / n_sweeps:.2f}s/sweep. Proceed to full simulation "
              "validation (ROADMAP.md Section 1, next item).")

    np.savez(
        args.out,
        alm_samples=samples, phi_samples=phi_samples, logp=logp, accepts=accepts,
        cl_true=cl_true, cl_phiphi_true=cl_phiphi_true,
        alm_true_packed=alm_true_packed, phi_true_packed=phi_true_packed,
        seconds_per_sweep=elapsed / n_sweeps,
    )
    print(f"\nSaved chain + truth to {args.out}")


if __name__ == "__main__":
    main()
