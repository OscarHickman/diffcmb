"""
ROADMAP.md Section 1, Phase 2's "CMBLensing.jl benchmark" item: the mandatory
first step before that benchmark -- a patch-scale (f_sky~0.016) sanity check
of the 3-block Gibbs sampler at production lmax.

All Phase 2 mixing/validation to date (gates 1/2, the lmax=300 smoke run, the
simulation-validation run) has used f_sky~0.7 or full-sky. Masking
pathologies (mode coupling from the sharp mask edge, near-degenerate
alm/phi directions concentrated in a small patch) are expected to grow as
the mask deepens -- this is the same concern Millea+2020's flat-sky work
was built around, but flat-sky patches ARE small-f_sky by construction, so
their mixing results don't directly bound ours. This script directly
measures it, at the same production lmax=300 scale the real benchmark and
Phase 2 chains will run at.

Masked-sky matrix-free lensing correctness was validated 2026-07-18
(tests/test_lensing.py, f_sky~0.3) -- this gate is about *mixing*, not
operator correctness, and is expected to be cheap because the ducc0
full-sky synthesis cost (the dominant per-sweep cost) doesn't scale down
with n_unmasked pixels, so there's no throughput reason to defer it.

Method
------
1. Build a matrix-free-SHT model, then override model.unmasked_idx with a
   small contiguous polar-cap mask (f_sky~0.016), following the same
   construction as tests/test_lensing.py::_polar_cap_mask_idx and
   tests/test_samplers.py's small_masked_matrixfree_model fixture.
2. Draw (alm_true, phi_true) from CAMB spectra, lens through the model's own
   matrix-free forward operator (lens_map_tf already returns only the
   n_unmasked pixel values for a masked model), add noise -- the "data".
3. Run the 3-block Gibbs chain (C_l|alm exact; alm|C_l,phi HMC; phi|alm,C_l
   HMC), warm-started at the truth so the diagnostic isolates mixing from
   burn-in-to-mode time (as in gates 1/2).
4. Compute IAT/ESS (gate 1's Sokal-windowed estimator) for alm-power and
   phi-power summary statistics at a spread of probed multipoles, and apply
   gate 1's same PATHOLOGICAL / MARGINAL / ACCEPTABLE thresholds.

If PATHOLOGICAL: the CMBLensing.jl benchmark design should be inverted
(full-sky vs QE as primary, per ROADMAP.md) rather than attempted at this
mask depth.

Cost note: NOT yet run. Per-sweep cost is expected close to the full-sky
lmax=300 pivot's ~29-54s/sweep (matrix-free full-sky synthesis doesn't get
cheaper for a smaller mask), so keep n_burnin/n_samples modest for a first
pass and rely on checkpointing (as validate_sim_lmax300_lensing.py does) if
a longer run turns out to be needed. Confirm the SLURM job count against
the durham 200-job cap before submitting (see ~/.claude/CLAUDE.md).

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/gate_patch_mixing_check.py \
    --lmax 300 --nside 256 --f_sky 0.016 --n_burnin 50 --n_samples 150 \
    --n_lfs 10 --phi_n_lfs 10 \
    --checkpoint_path results/analysis/gate_patch_mixing_ckpt.npz
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


def polar_cap_mask_idx(nside, f_sky):
    """Contiguous polar-cap mask covering the requested f_sky fraction
    (same construction as tests/test_lensing.py::_polar_cap_mask_idx)."""
    npix = 12 * nside * nside
    theta, _ = hp.pix2ang(nside, np.arange(npix))
    cutoff = np.arccos(1 - 2 * f_sky)
    return np.where(theta < cutoff)[0]


def integrated_autocorr_time(x, c=5.0):
    """Sokal's windowed IAT estimator (as in gate_phi_alm_mixing_check.py)."""
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
    p.add_argument("--lmax", type=int, default=300)
    p.add_argument("--nside", type=int, default=256)
    p.add_argument("--f_sky", type=float, default=0.016)
    p.add_argument("--noisesig", type=float, default=1.0)
    p.add_argument("--n_burnin", type=int, default=50)
    p.add_argument("--n_samples", type=int, default=150)
    p.add_argument("--hmc_step_size", type=float, default=0.01)
    p.add_argument("--n_lfs", type=int, default=10)
    p.add_argument("--phi_hmc_step_size", type=float, default=0.001)
    p.add_argument("--phi_n_lfs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint_path", type=str,
                    default="results/analysis/gate_patch_mixing_ckpt.npz")
    p.add_argument("--checkpoint_every", type=int, default=25)
    p.add_argument("--out", type=str, default="results/analysis/gate_patch_mixing_check.npz")
    args = p.parse_args()

    lmax, nside = args.lmax, args.nside
    rng = np.random.default_rng(args.seed)

    print(f"=== Patch-scale (f_sky={args.f_sky}) mixing sanity check "
          f"(lmax={lmax}, nside={nside}) ===\n")

    print("Building matrix-free-SHT model, then overriding to a small polar-cap mask...")
    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=args.noisesig,
        data_mode="synthetic", dtype=tf.complex128, use_matrixfree_sht=True,
    )
    model.unmasked_idx = polar_cap_mask_idx(nside, args.f_sky)
    print(f"  unmasked pixels: {len(model.unmasked_idx)} / {model.NPIX} "
          f"(f_sky={len(model.unmasked_idx) / model.NPIX:.4f})")
    model._ensure_tf_tensors()

    print("Drawing (alm_true, phi_true) from CAMB spectra at the fixed LCDM cosmology...")
    cl_true = call_CAMB_map(LCDM_PARAMS, lmax)
    cl_phiphi_true = get_cl_phiphi(lmax)

    alm_true_hp = hp.synalm(cl_true, lmax=lmax - 1, new=True).astype(np.complex128)
    phi_true_hp = hp.synalm(cl_phiphi_true, lmax=lmax - 1, new=True).astype(np.complex128)
    alm_true_packed = _alm_hp_to_packed(alm_true_hp, lmax)
    phi_true_packed = _alm_hp_to_packed(phi_true_hp, lmax)

    print("Lensing the true sky through the matrix-free forward operator and adding noise "
          "(lens_map_tf on a masked model returns only the unmasked-pixel values)...")
    T_lensed_true_masked = lens_map_tf(
        model, tf.constant(alm_true_packed, tf.float64), phi_true_hp
    ).numpy()
    if not np.all(np.isfinite(T_lensed_true_masked)):
        raise RuntimeError("lens_map_tf produced NaN/Inf -- aborting gate")

    noisy_masked = T_lensed_true_masked + rng.normal(
        0.0, args.noisesig, size=len(model.unmasked_idx)
    )
    model.prior_map_masked = tf.convert_to_tensor(noisy_masked, dtype=tf.float64)

    x0 = np.concatenate([np.log(cl_true[2:lmax]), alm_true_packed])

    print(f"\nRunning 3-block Gibbs chain: n_burnin={args.n_burnin}, "
          f"n_samples={args.n_samples}, n_lfs={args.n_lfs}, phi_n_lfs={args.phi_n_lfs}, "
          f"warm-started at truth (checkpoint={args.checkpoint_path})...")
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
    print(f"  done in {elapsed / 60:.1f} min ({elapsed / max(1, len(samples)):.2f}s/collected-sample)")
    print(f"  alm-block accept rate: {accepts.mean():.3f}")

    if not (np.all(np.isfinite(samples)) and np.all(np.isfinite(phi_samples))
            and np.all(np.isfinite(logp))):
        raise RuntimeError("NaN/Inf in samples/phi_samples/logp -- gate FAILED")

    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    L_arr, _m_arr = _alm_index_lm(lmax, n_real, n_imag)

    alm_part = samples[:, n_lncl:]

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
        print("PATHOLOGICAL: deep masking breaks mixing at this scale. Invert the "
              "CMBLensing.jl benchmark design (full-sky vs QE as primary) rather than "
              "attempting a matched flat-patch comparison (ROADMAP.md Section 1).")
    elif worst_frac < 10.0:
        print("MARGINAL: mixing is slow but not catastrophic -- consistent with gate 1's "
              "full-sky-scale result; proceed to the CMBLensing.jl benchmark with this "
              "caveat noted, or re-check with more samples first.")
    else:
        print("ACCEPTABLE: no evidence of mask-depth-driven mixing pathology. Proceed to "
              "the CMBLensing.jl matched flat-patch benchmark.")

    np.savez(
        args.out,
        alm_samples=samples, phi_samples=phi_samples, logp=logp, accepts=accepts,
        cl_true=cl_true, cl_phiphi_true=cl_phiphi_true,
        alm_true_packed=alm_true_packed, phi_true_packed=phi_true_packed,
        unmasked_idx=model.unmasked_idx, f_sky=args.f_sky,
        probe_ells=probe_ells, seconds_total=elapsed,
    )
    print(f"\nSaved chain + truth to {args.out}")


if __name__ == "__main__":
    main()
