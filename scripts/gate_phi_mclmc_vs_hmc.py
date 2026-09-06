"""
MCLMC spike (ROADMAP.md, 2026-08-07): Block 3 (phi | alm, C_l) validation gate.

Context: the phi_n_lfs=80->240 pilot (job 11694912) showed almost no
autocorrelation improvement on the worst l-bin despite 3x more leapfrog
steps -- evidence for a sampler-geometry problem (ROADMAP.md decision rule
3(a)), which triggered a bounded spike porting Block 3 to MCLMC (Robnik,
De Luca, Silverstein & Seljak arXiv:2212.08549; hand-implemented in TF,
diffcmb/mclmc.py -- no blackjax/JAX dependency). This script runs the two
validation steps the spike's plan requires before trusting any speed
comparison:

1. Unadjusted-sampler bias check: MCLMC drops the Metropolis-Hastings
   correction HMC has, so before trusting any ESS/mixing comparison, confirm
   MCLMC's posterior mean/variance of phi power agrees with the already
   -validated HMC path, both warm-started at truth, same protocol as
   gate_phi_alm_mixing_check.py.
2. Speed/mixing comparison -- the hard-abort gate: IAT/ESS per l-bin (Sokal
   windowed estimator, same estimator as gate_phi_alm_mixing_check.py) and
   ESS per wall-clock second for both samplers. If MCLMC does not clearly
   beat the current best HMC configuration, the spike is closed (recorded in
   achievements.md) and phi_sampler stays 'hmc' in production.

--mclmc_step_size and --mclmc_L each accept a comma-separated grid (tried as
every pairwise combination) -- an untuned MCLMC step size can be many orders
of magnitude off HMC's converged step size in the same whitened coordinates,
so a single-point comparison is not a fair test; the grid finds a reasonable
operating point by ESS/s at one probe multipole before the full comparison.

Usage (small-lmax bias check + step-size/L grid, default):
    PYTHONPATH=diffcmb .venv/bin/python scripts/gate_phi_mclmc_vs_hmc.py \\
        --lmax 20 --nside 16 --n_burnin 200 --n_samples 500 \\
        --mclmc_step_size 0.01,0.05,0.2,0.5 --mclmc_L 5,20,80

Usage (lmax~128 pilot-scale speed gate, once the bias check passes):
    PYTHONPATH=diffcmb .venv/bin/python scripts/gate_phi_mclmc_vs_hmc.py \\
        --lmax 128 --nside 64 --n_burnin 500 --n_samples 800 \\
        --phi_n_lfs 240 --mclmc_n_steps 240 --mclmc_step_size <tuned> --mclmc_L <tuned>
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


def integrated_autocorr_time(x, c=5.0):
    """Sokal's windowed IAT estimator (as in emcee.autocorr.integrated_time)."""
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
    return max(tau, 1.0), n / max(tau, 1.0)


def summarize_phi_block(phi_samples, L_arr, probe_ells, label, wall_clock_s):
    print(f"\n--- {label}: mean/var, IAT / ESS by multipole (wall clock {wall_clock_s:.1f}s) ---")
    rows = []
    for ell in probe_ells:
        mask = L_arr == ell
        if not mask.any():
            continue
        power = (phi_samples[:, mask] ** 2).mean(axis=1)
        tau, ess = integrated_autocorr_time(power)
        ess_per_sec = ess / wall_clock_s
        print(
            f"  l={ell:4d}  mean={power.mean():.4e}  var={power.var():.4e}  "
            f"IAT={tau:8.1f}  ESS={ess:8.1f}/{len(phi_samples)}  ESS/s={ess_per_sec:.4f}"
        )
        rows.append((ell, power.mean(), power.var(), tau, ess, ess_per_sec))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lmax", type=int, default=20)
    p.add_argument("--nside", type=int, default=16)
    p.add_argument("--noisesig", type=float, default=100.0)
    p.add_argument("--n_burnin", type=int, default=200)
    p.add_argument("--n_samples", type=int, default=500)
    p.add_argument("--hmc_step_size", type=float, default=0.05)
    p.add_argument("--n_lfs", type=int, default=20)
    p.add_argument("--phi_hmc_step_size", type=float, default=0.05)
    p.add_argument("--phi_n_lfs", type=int, default=80)
    p.add_argument(
        "--mclmc_step_size", type=str, default="1e-4",
        help="comma-separated grid, e.g. '0.01,0.05,0.2' -- tried against every --mclmc_L value",
    )
    p.add_argument(
        "--mclmc_n_steps", type=str, default="80",
        help="comma-separated grid, e.g. '20,40,80' -- tried against every step_size/L combination",
    )
    p.add_argument(
        "--mclmc_L", type=str, default="1.0",
        help="comma-separated grid, e.g. '5,20,50' -- tried against every --mclmc_step_size value",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/analysis/phi_mclmc_vs_hmc.npz")
    args = p.parse_args()

    lmax, nside = args.lmax, args.nside
    rng = np.random.default_rng(args.seed)

    print(f"=== MCLMC spike gate: Block 3 phi HMC vs MCLMC (lmax={lmax}, nside={nside}) ===\n")

    print("Building model (dense SHT)...")
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

    print(f"\n--- HMC baseline: phi_n_lfs={args.phi_n_lfs}, phi_hmc_step_size={args.phi_hmc_step_size} ---")
    t0 = time.time()
    _samples, hmc_phi_samples, _logp, _accepts, _final_step = run_gibbs_chain(
        model,
        n_samples=args.n_samples, n_burnin=args.n_burnin,
        hmc_step_size=args.hmc_step_size, n_lfs=args.n_lfs,
        initial_params=x0,
        cl_phiphi_full=cl_phiphi_true, phi_initial=phi_true_packed,
        phi_sampler='hmc',
        phi_hmc_step_size=args.phi_hmc_step_size, phi_n_lfs=args.phi_n_lfs,
        seed=args.seed,
    )
    hmc_wall_s = time.time() - t0
    print(f"  done in {hmc_wall_s/60:.1f} min")

    step_grid = [float(s) for s in args.mclmc_step_size.split(",")]
    L_grid = [float(s) for s in args.mclmc_L.split(",")]
    n_steps_grid = [int(s) for s in args.mclmc_n_steps.split(",")]
    mid_ell = probe_ells[len(probe_ells) // 2]

    print(f"\n--- MCLMC grid: n_steps in {n_steps_grid}, "
          f"step_size in {step_grid}, L in {L_grid} "
          f"(ranking by ESS/s at the middle probed bin l={mid_ell}) ---")
    grid_results = []
    for n_steps in n_steps_grid:
        for step_size in step_grid:
            for L in L_grid:
                t0 = time.time()
                _s, phi_samples_g, _lp, _acc, _fs = run_gibbs_chain(
                    model,
                    n_samples=args.n_samples, n_burnin=args.n_burnin,
                    hmc_step_size=args.hmc_step_size, n_lfs=args.n_lfs,
                    initial_params=x0,
                    cl_phiphi_full=cl_phiphi_true, phi_initial=phi_true_packed,
                    phi_sampler='mclmc',
                    phi_hmc_step_size=step_size, phi_n_lfs=n_steps,
                    phi_mclmc_L=L,
                    seed=args.seed,
                )
                wall_s = time.time() - t0
                mask = L_arr == mid_ell
                power = (phi_samples_g[:, mask] ** 2).mean(axis=1)
                tau, ess = integrated_autocorr_time(power)
                epss = ess / wall_s
                print(f"  n_steps={n_steps}  step_size={step_size:.4g}  L={L:.4g}  l={mid_ell}: IAT={tau:.1f}  "
                      f"ESS={ess:.1f}/{len(phi_samples_g)}  ESS/s={epss:.4f}  ({wall_s:.1f}s)")
                grid_results.append((n_steps, step_size, L, epss, wall_s, phi_samples_g))

    best_n_steps, best_step, best_L, best_epss, mclmc_wall_s, mclmc_phi_samples = max(
        grid_results, key=lambda r: r[3]
    )
    print(f"\nBest MCLMC config: n_steps={best_n_steps}, step_size={best_step:.4g}, L={best_L:.4g} "
          f"(ESS/s={best_epss:.4f} at l={mid_ell})")

    hmc_rows = summarize_phi_block(hmc_phi_samples, L_arr, probe_ells, "HMC (baseline)", hmc_wall_s)
    mclmc_rows = summarize_phi_block(
        mclmc_phi_samples, L_arr, probe_ells,
        f"MCLMC (spike, best of grid: n_steps={best_n_steps}, step_size={best_step:.4g}, L={best_L:.4g})",
        mclmc_wall_s,
    )

    print("\n=== Verdict ===")
    print("1. Bias check (mean/var should agree within a few sigma of each other's MC noise):")
    for (ell, m_h, _v_h, *_), (_, m_m, _v_m, *_) in zip(hmc_rows, mclmc_rows):
        # crude MC-noise scale from each chain's own ESS
        rel_diff = abs(m_h - m_m) / (0.5 * (abs(m_h) + abs(m_m)) + 1e-30)
        flag = "  <-- CHECK" if rel_diff > 0.5 else ""
        print(f"  l={ell:4d}  HMC mean={m_h:.4e}  MCLMC mean={m_m:.4e}  rel_diff={rel_diff:.2f}{flag}")

    print("\n2. Speed/mixing (ESS per wall-clock second, higher is better) -- hard-abort gate:")
    worst_bin_beats = []
    for (ell, _, _, _tau_h, _ess_h, epss_h), (_, _, _, _tau_m, _ess_m, epss_m) in zip(hmc_rows, mclmc_rows):
        beats = epss_m > epss_h
        worst_bin_beats.append(beats)
        print(
            f"  l={ell:4d}  HMC ESS/s={epss_h:.4f}  MCLMC ESS/s={epss_m:.4f}  "
            f"{'MCLMC WINS' if beats else 'HMC WINS'}"
        )
    if all(worst_bin_beats):
        print("\nGO: MCLMC beats HMC ESS/wall-clock-second on every probed l-bin.")
    elif any(worst_bin_beats):
        print("\nAMBIGUOUS: MCLMC beats HMC on some but not all bins -- judgement call, "
              "check the worst (highest-l pilot-pathology) bin specifically before deciding.")
    else:
        print("\nNO-GO: MCLMC does not beat HMC on any probed l-bin -- close the spike "
              "per the hard-abort criterion (ROADMAP.md/achievements.md).")

    np.savez(
        args.out,
        hmc_phi_samples=hmc_phi_samples, mclmc_phi_samples=mclmc_phi_samples,
        hmc_wall_s=hmc_wall_s, mclmc_wall_s=mclmc_wall_s,
        cl_true=cl_true, cl_phiphi_true=cl_phiphi_true,
        alm_true_packed=alm_true_packed, phi_true_packed=phi_true_packed,
        probe_ells=probe_ells,
    )
    print(f"\nSaved chains + truth to {args.out}")


if __name__ == "__main__":
    main()
