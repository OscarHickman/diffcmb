"""
ROADMAP.md Section 1, "Simulation validation" follow-up: local smoke test for
the Fisher-informed phi HMC mass matrix (samplers.py::build_phi_posterior_mass_sqrt,
lensing.py::estimate_phi_diag_fisher).

Job 11612969 (validate_sim_lmax300_lensing.py, lmax=300) completed but failed
its point-agreement check; post-hoc diagnosis found the phi block's prior-only
mass matrix (build_phi_prior_mass_sqrt) leaves it barely mixing -- lag-1/lag-5
autocorrelation of phi power >0.9999, a smooth near-monotonic drift rather
than noise around a mean (see achievements.md). This script checks, at a cheap
small-lmax scale, whether phi_mass_matrix='fisher' actually improves that
before spending another multi-hour SLURM job on a re-run.

Method
------
1. Build a small dense-SHT model, draw (alm_true, phi_true) from CAMB spectra,
   lens through the model's own forward operator, add noise -- the "data"
   (same construction as gate_phi_alm_mixing_check.py).
2. Run two short 3-block Gibbs chains, warm-started at the truth so the
   diagnostic isolates mixing, not burn-in-to-mode time: one with
   phi_mass_matrix='prior' (current default), one with 'fisher'.
3. Compare lag-1 autocorrelation of the phi power at a spread of probed
   multipoles between the two runs -- 'fisher' should be substantially lower
   at low-L (likelihood-dominated) bins, where the prior-only mass matrix is
   most wrong.

This is a local diagnostic, not a SLURM gate -- run directly:
    PYTHONPATH=diffcmb .venv/bin/python scripts/smoke_phi_fisher_mass_matrix.py \
        --lmax 20 --nside 16 --n_burnin 50 --n_samples 300
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
    """CAMB lensing-potential power spectrum C_L^phiphi -- same construction
    as gate_phi_alm_mixing_check.py::get_cl_phiphi."""
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


def lag1_autocorr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if np.allclose(x, 0.0):
        return 1.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def report_lag1(phi_samples, L_arr, probe_ells, label):
    print(f"\n--- {label}: lag-1 autocorrelation of phi power by multipole ---")
    rows = []
    for ell in probe_ells:
        mask = L_arr == ell
        if not mask.any():
            continue
        power = (phi_samples[:, mask] ** 2).mean(axis=1)
        ac1 = lag1_autocorr(power)
        print(f"  l={ell:4d}  lag1_autocorr={ac1:7.4f}")
        rows.append((ell, ac1))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lmax", type=int, default=20)
    p.add_argument("--nside", type=int, default=16)
    p.add_argument("--noisesig", type=float, default=1.0)
    p.add_argument("--n_burnin", type=int, default=50)
    p.add_argument("--n_samples", type=int, default=300)
    p.add_argument("--hmc_step_size", type=float, default=0.05)
    p.add_argument("--n_lfs", type=int, default=20)
    p.add_argument("--phi_hmc_step_size", type=float, default=0.05)
    p.add_argument("--phi_n_lfs", type=int, default=20)
    p.add_argument("--phi_fisher_warmup_iter", type=int, default=10)
    p.add_argument("--phi_fisher_n_probes", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    lmax, nside = args.lmax, args.nside
    rng = np.random.default_rng(args.seed)

    print(f"=== phi Fisher mass-matrix smoke test (lmax={lmax}, nside={nside}) ===\n")

    print("Building model (dense SHT -- Block 3 requires model.sph_parts)...")
    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=args.noisesig,
        data_mode="synthetic", dtype=tf.complex128,
    )
    model._ensure_tf_tensors()

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

    x0 = np.concatenate([np.log(cl_true[2:lmax]), alm_true_packed])
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    L_arr, _m_arr = _alm_index_lm(lmax, n_real, n_imag)
    probe_ells = sorted({max(2, min(lmax - 1, v)) for v in
                          np.linspace(2, lmax - 1, 6).round().astype(int)})

    results = {}
    for mode in ("prior", "fisher"):
        print(f"\nRunning 3-block Gibbs chain: phi_mass_matrix={mode!r}, "
              f"n_burnin={args.n_burnin}, n_samples={args.n_samples}, warm-started at truth...")
        t0 = time.time()
        _samples, phi_samples, _logp, accepts, _final_step = run_gibbs_chain(
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
            phi_mass_matrix=mode,
            phi_fisher_warmup_iter=args.phi_fisher_warmup_iter,
            phi_fisher_n_probes=args.phi_fisher_n_probes,
            seed=args.seed,
        )
        print(f"  done in {(time.time()-t0)/60:.1f} min, alm accept={accepts.mean():.3f}")
        results[mode] = report_lag1(phi_samples, L_arr, probe_ells, f"phi_mass_matrix={mode!r}")

    print("\n=== Verdict ===")
    improved, worse = 0, 0
    for (ell, ac_prior), (_ell2, ac_fisher) in zip(results["prior"], results["fisher"]):
        delta = ac_prior - ac_fisher
        tag = "IMPROVED" if delta > 0.01 else ("WORSE" if delta < -0.01 else "~same")
        if tag == "IMPROVED":
            improved += 1
        elif tag == "WORSE":
            worse += 1
        print(f"  l={ell:4d}  prior={ac_prior:7.4f}  fisher={ac_fisher:7.4f}  {tag}")
    print(f"\n{improved} multipole(s) improved, {worse} worse, "
          f"{len(results['prior']) - improved - worse} unchanged.")
    if improved > worse:
        print("Fisher mass matrix reduces phi autocorrelation at this scale -- "
              "worth trying at lmax=300 before spending another SLURM job.")
    else:
        print("Fisher mass matrix does not clearly help at this scale -- "
              "investigate further before scaling up.")


if __name__ == "__main__":
    main()
