"""
ROADMAP.md Section 1, "Simulation validation" follow-up: local smoke test for
whether a longer phi-block leapfrog schedule (not a new mass matrix) fixes the
near-frozen phi mixing found in job 11612969 (achievements.md).

The Fisher-informed mass matrix route (samplers.py::build_phi_posterior_mass_sqrt)
was tried and found unfavorable on average across 5 seeds (see achievements.md,
"Fisher-informed phi HMC mass matrix"). This script tests the other half of the
open question in ROADMAP.md: at production scale the adapted phi step size
settled very small (~6.4e-4 at lmax=300), so the HMC trajectory length
(step_size * num_leapfrog_steps) may simply be too short to move the chain,
independent of preconditioning. If so, raising phi_n_lfs at fixed
phi_mass_matrix='prior' should reduce phi power autocorrelation without
touching the mass matrix at all.

Method
------
Same synthetic setup as smoke_phi_fisher_mass_matrix.py (warm-started at
truth, dense-SHT small model): run several short 3-block Gibbs chains with
phi_mass_matrix='prior' fixed and phi_n_lfs swept across a range, and compare
lag-1 autocorrelation of phi power across probed multipoles.

Local diagnostic, not a SLURM gate -- run directly:
    PYTHONPATH=diffcmb .venv/bin/python scripts/smoke_phi_leapfrog_schedule.py \
        --lmax 12 --nside 8 --n_burnin 50 --n_samples 300 --seeds 0,1,2,3,4
"""
import argparse

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


def lag1_autocorr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if np.allclose(x / max(np.abs(x).max(), 1e-300), 0.0):
        return 1.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def run_one_seed(lmax, nside, noisesig, n_burnin, n_samples, hmc_step_size, n_lfs,
                  phi_hmc_step_size, phi_n_lfs_values, seed, probe_ells):
    rng = np.random.default_rng(seed)

    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=noisesig,
        data_mode="synthetic", dtype=tf.complex128,
    )
    model._ensure_tf_tensors()

    cl_true = call_CAMB_map(LCDM_PARAMS, lmax)
    cl_phiphi_true = get_cl_phiphi(lmax)

    alm_true_hp = hp.synalm(cl_true, lmax=lmax - 1, new=True).astype(np.complex128)
    phi_true_hp = hp.synalm(cl_phiphi_true, lmax=lmax - 1, new=True).astype(np.complex128)
    alm_true_packed = _alm_hp_to_packed(alm_true_hp, lmax)
    phi_true_packed = _alm_hp_to_packed(phi_true_hp, lmax)

    T_lensed_true = lens_map_tf(
        model, tf.constant(alm_true_packed, tf.float64), phi_true_hp
    ).numpy()
    noisy_map = T_lensed_true + rng.normal(0.0, noisesig, size=model.NPIX)
    model.prior_map = noisy_map
    model.prior_map_parts = [tf.convert_to_tensor(noisy_map[model.unmasked_idx], dtype=tf.float64)]

    x0 = np.concatenate([np.log(cl_true[2:lmax]), alm_true_packed])

    seed_results = {}
    for phi_n_lfs in phi_n_lfs_values:
        _samples, phi_samples, _logp, accepts, _final_step = run_gibbs_chain(
            model,
            n_samples=n_samples,
            n_burnin=n_burnin,
            hmc_step_size=hmc_step_size,
            n_lfs=n_lfs,
            initial_params=x0,
            cl_phiphi_full=cl_phiphi_true,
            phi_initial=phi_true_packed,
            phi_hmc_step_size=phi_hmc_step_size,
            phi_n_lfs=phi_n_lfs,
            phi_mass_matrix="prior",
            seed=seed,
        )
        n_real = lmax * (lmax + 1) // 2 - 3
        n_imag = packed_sizes(lmax)[1]
        L_arr, _m_arr = _alm_index_lm(lmax, n_real, n_imag)
        rows = []
        for ell in probe_ells:
            mask = L_arr == ell
            if not mask.any():
                continue
            power = (phi_samples[:, mask] ** 2).mean(axis=1)
            rows.append((ell, lag1_autocorr(power)))
        seed_results[phi_n_lfs] = rows
        print(f"    seed={seed} phi_n_lfs={phi_n_lfs}: alm accept={accepts.mean():.3f} "
              + " ".join(f"l={ell}:{ac:.3f}" for ell, ac in rows))
    return seed_results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lmax", type=int, default=12)
    p.add_argument("--nside", type=int, default=8)
    p.add_argument("--noisesig", type=float, default=1.0)
    p.add_argument("--n_burnin", type=int, default=50)
    p.add_argument("--n_samples", type=int, default=300)
    p.add_argument("--hmc_step_size", type=float, default=0.05)
    p.add_argument("--n_lfs", type=int, default=20)
    p.add_argument("--phi_hmc_step_size", type=float, default=0.05)
    p.add_argument("--phi_n_lfs_values", type=str, default="20,80,200")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    args = p.parse_args()

    phi_n_lfs_values = [int(v) for v in args.phi_n_lfs_values.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    probe_ells = sorted({max(2, min(args.lmax - 1, v)) for v in
                          np.linspace(2, args.lmax - 1, 4).round().astype(int)})

    print(f"=== phi leapfrog-schedule smoke test (lmax={args.lmax}, nside={args.nside}, "
          f"phi_n_lfs in {phi_n_lfs_values}, seeds={seeds}) ===\n")

    all_results = {n: [] for n in phi_n_lfs_values}
    for seed in seeds:
        seed_results = run_one_seed(
            args.lmax, args.nside, args.noisesig, args.n_burnin, args.n_samples,
            args.hmc_step_size, args.n_lfs, args.phi_hmc_step_size,
            phi_n_lfs_values, seed, probe_ells,
        )
        for n, rows in seed_results.items():
            all_results[n].append(rows)

    print("\n=== Verdict (mean lag-1 autocorrelation across seeds, per phi_n_lfs) ===")
    baseline = phi_n_lfs_values[0]
    means = {}
    for n in phi_n_lfs_values:
        by_ell = {}
        for rows in all_results[n]:
            for ell, ac in rows:
                by_ell.setdefault(ell, []).append(ac)
        mean_all = np.mean([ac for rows in all_results[n] for _, ac in rows])
        means[n] = mean_all
        per_ell_str = " ".join(f"l={ell}:{np.mean(v):.4f}" for ell, v in sorted(by_ell.items()))
        print(f"  phi_n_lfs={n:4d}  mean_autocorr={mean_all:.4f}  ({per_ell_str})")

    print()
    for n in phi_n_lfs_values[1:]:
        delta = means[baseline] - means[n]
        tag = "IMPROVED" if delta > 0.01 else ("WORSE" if delta < -0.01 else "~same")
        print(f"  phi_n_lfs={n} vs baseline={baseline}: delta={delta:+.4f} ({tag})")


if __name__ == "__main__":
    main()
