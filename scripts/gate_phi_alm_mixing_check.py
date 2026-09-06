"""
ROADMAP.md Section 1, Phase 2 pre-production gate 1: joint (alm, phi) mixing
check at lmax<=50.

Millea+2020's central methodological contribution was a reparametrisation
introduced precisely because naive block alternation (alm | phi, C_l then
phi | alm, C_l, in turn) mixes catastrophically at high signal-to-noise --
the two blocks trade off along a near-degenerate direction that neither HMC
step explores efficiently alone. This script measures whether the *already
implemented* Block 2/Block 3 alternation (run_gibbs_chain with
cl_phiphi_full set, see samplers.py) shows that pathology, directly, on a
synthetic lensed sky with known (alm_true, phi_true) at a cheap lmax.

Method
------
1. Build a small dense-SHT model (this gate predates the matrix-free port of
   Block 3 -- lensing.py's psi_lensed/lens_map_tf now also support
   use_matrixfree_sht=True, see ROADMAP.md Section 1 and lensing.py).
2. Draw alm_true ~ N(0, C_l^TT) and phi_true ~ N(0, C_l^phiphi) from CAMB
   spectra at the model's fixed LCDM cosmology (power.py's parameters).
3. Lens alm_true by phi_true through the model's own forward operator
   (lens_map_tf, dense-SHT-consistent) and add noise -- the "data".
4. Run the 3-block Gibbs chain (C_l | alm exact; alm | C_l, phi HMC;
   phi | alm, C_l HMC), warm-started at the truth so the diagnostic isolates
   mixing, not burn-in-to-mode time.
5. Compute the integrated autocorrelation time (IAT, Sokal's windowed
   estimator) of alm-power and phi-power summary statistics at a spread of
   probed multipoles. High IAT relative to n_samples (low effective sample
   size) is the direct symptom Millea+2020 describes.

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/gate_phi_alm_mixing_check.py \
    --lmax 50 --nside 64 --n_burnin 500 --n_samples 3000
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
    """CAMB lensing-potential power spectrum C_L^phiphi at the model's
    fixed LCDM cosmology (power.py's call_CAMB_map uses the same
    parameters for C_l^TT)."""
    import camb

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=LCDM_PARAMS[0], ombh2=LCDM_PARAMS[1], omch2=LCDM_PARAMS[2],
        mnu=LCDM_PARAMS[3], omk=LCDM_PARAMS[4], tau=LCDM_PARAMS[5],
    )
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    results = camb.get_results(pars)
    # get_lens_potential_cls returns dimensionless [L(L+1)]^2 C_L^pp / 2pi
    # in column 0 (pp), by default up to lmax.
    dl_pp = results.get_lens_potential_cls(lmax=lmax - 1)[:, 0]
    cl_pp = np.zeros(lmax, dtype=np.float64)
    for ell in range(2, lmax):
        if ell < len(dl_pp):
            norm = (ell * (ell + 1)) ** 2 / (2.0 * np.pi)
            cl_pp[ell] = dl_pp[ell] / norm
    return cl_pp


def integrated_autocorr_time(x, c=5.0):
    """Sokal's windowed IAT estimator (as in emcee.autocorr.integrated_time).

    x : 1-D array of a scalar summary statistic, one value per sweep.
    Returns (tau, ess) where ess = len(x) / tau. Falls back to tau=len(x)
    (ess=1) if the automatic window never converges (chain too short/too
    correlated to trust any window).
    """
    n = len(x)
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    scale = np.abs(x).max()
    if scale == 0.0 or np.allclose(x / scale, 0.0):
        return 1.0, float(n)

    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    acf /= acf[0]

    tau = 1.0
    for window in range(1, n):
        tau = 1.0 + 2.0 * acf[1:window + 1].sum()
        if window >= c * tau:
            return max(tau, 1.0), n / max(tau, 1.0)
    # Never converged -- chain is shorter than ~c * tau; report the
    # longest-window estimate as a (conservative, likely underestimated) tau.
    return max(tau, 1.0), n / max(tau, 1.0)


def summarize_block(samples, L_arr, probe_ells, label):
    print(f"\n--- {label}: IAT / ESS by multipole ---")
    rows = []
    for ell in probe_ells:
        mask = L_arr == ell
        if not mask.any():
            continue
        power = (samples[:, mask] ** 2).mean(axis=1)
        tau, ess = integrated_autocorr_time(power)
        frac = 100.0 * ess / len(samples)
        print(f"  l={ell:4d}  IAT={tau:8.1f}  ESS={ess:8.1f}/{len(samples)}  ({frac:5.1f}%)")
        rows.append((ell, tau, ess))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lmax", type=int, default=50)
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--noisesig", type=float, default=1.0)
    p.add_argument("--n_burnin", type=int, default=500)
    p.add_argument("--n_samples", type=int, default=3000)
    p.add_argument("--hmc_step_size", type=float, default=0.05)
    p.add_argument("--n_lfs", type=int, default=20)
    p.add_argument("--phi_hmc_step_size", type=float, default=0.05)
    p.add_argument("--phi_n_lfs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/analysis/phi_alm_mixing_check.npz")
    args = p.parse_args()

    lmax, nside = args.lmax, args.nside
    rng = np.random.default_rng(args.seed)

    print(f"=== Phase 2 pre-production gate 1: joint (alm, phi) mixing check "
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

    print(f"\nRunning 3-block Gibbs chain: n_burnin={args.n_burnin}, "
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
    print(f"  done in {(time.time()-t0)/60:.1f} min")
    print(f"  alm-block accept rate: {accepts.mean():.3f}")

    alm_part = samples[:, n_lncl:]
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    L_arr, _m_arr = _alm_index_lm(lmax, n_real, n_imag)

    probe_ells = sorted({max(2, min(lmax - 1, v)) for v in
                          np.linspace(2, lmax - 1, 6).round().astype(int)})

    alm_rows = summarize_block(alm_part, L_arr, probe_ells, "alm block (Block 2)")
    phi_rows = summarize_block(phi_samples, L_arr, probe_ells, "phi block (Block 3)")

    alm_frac = np.array([r[2] for r in alm_rows]) / args.n_samples
    phi_frac = np.array([r[2] for r in phi_rows]) / args.n_samples
    worst_frac = min(alm_frac.min(), phi_frac.min()) * 100.0

    print("\n=== Verdict ===")
    print(f"Worst ESS fraction across both blocks: {worst_frac:.2f}%")
    if worst_frac < 1.0:
        print("PATHOLOGICAL: consistent with Millea+2020's naive-block-alternation "
              "failure mode. Escalate to a reparametrisation or joint (alm,phi) HMC "
              "before any lmax=300 Phase 2 chain (ROADMAP.md Section 1).")
    elif worst_frac < 10.0:
        print("MARGINAL: mixing is slow but not catastrophic -- more samples may "
              "suffice; re-check at lmax=50 with the intended production sample "
              "budget before scaling to lmax=300.")
    else:
        print("ACCEPTABLE: no evidence of catastrophic block-alternation mixing "
              "at this lmax/S-N. Gate 1 passes; proceed to gate 2 "
              "(Block-2-exactness experiment).")

    np.savez(
        args.out,
        alm_samples=samples, phi_samples=phi_samples, logp=logp, accepts=accepts,
        cl_true=cl_true, cl_phiphi_true=cl_phiphi_true,
        alm_true_packed=alm_true_packed, phi_true_packed=phi_true_packed,
        probe_ells=probe_ells,
    )
    print(f"\nSaved chain + truth to {args.out}")


if __name__ == "__main__":
    main()
