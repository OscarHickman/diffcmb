"""
Phase 0c Step 6: validate the production messenger-sampler chains at lmax=300.

Checks (per ROADMAP.md Phase 0c Step 6):
  - R-hat across the 4 messenger chains (Cl and alm blocks)
  - ESS per ln(Cl) coefficient, compared to the Phase 0 float64 HMC baseline
  - C_l posterior agreement with the Phase 0 float64 HMC reference chain (l <= 100)
  - logp trace sanity (no drift/divergence)
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from diffcmb import load_cmb_chains

OUT_DIR = "results/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

LMAX = 300
N_LNCL = LMAX - 2
MESSENGER_DIR = "results/lmax300_nside256_gibbs_real_messenger"
REFERENCE_DIR = "results/lmax300_nside256_gibbs_real_double"  # Phase 0 float64 HMC baseline
LCDM_PARAMS = [67.74, 0.0486, 0.2589, 0.06, 0.0, 0.066]


def gelman_rubin(chains):
    """R-hat per parameter across a list of (n_samples, n_params) arrays."""
    M = len(chains)
    N = min(c.shape[0] for c in chains)
    stacked = np.stack([c[:N] for c in chains], axis=0)  # (M, N, P)
    chain_means = stacked.mean(axis=1)
    grand_mean = chain_means.mean(axis=0)
    B = N / (M - 1) * ((chain_means - grand_mean) ** 2).sum(axis=0)
    W = (((stacked - chain_means[:, None, :]) ** 2).sum(axis=1) / (N - 1)).mean(axis=0)
    var_hat = (N - 1) / N * W + B / N
    return np.sqrt(var_hat / (W + 1e-30))


def compute_ess(chains):
    """ESS per parameter across a list of (n_samples, n_params) arrays, via TFP."""
    import tensorflow as tf  # noqa: F401
    import tensorflow_probability as tfp

    N = min(c.shape[0] for c in chains)
    stacked = np.stack([c[:N] for c in chains], axis=1)  # (N, M, P)
    # cross_chain_dims=1 combines ESS across the chain axis (Vehtari et al. 2021),
    # giving one ESS per parameter comparable to N_total = N * M.
    return tfp.mcmc.effective_sample_size(stacked, cross_chain_dims=1).numpy()


def main():
    print(f"=== Phase 0c Step 6 validation: messenger sampler, lmax={LMAX} ===\n")

    msgr_samples, msgr_logp, msgr_accept = load_cmb_chains(MESSENGER_DIR)
    n_chains = len(msgr_samples)
    if n_chains == 0:
        print(f"No chains found in {MESSENGER_DIR}")
        return

    for i, (ar, lp) in enumerate(zip(msgr_accept, msgr_logp, strict=True)):
        finite = lp[np.isfinite(lp)]
        print(f"  Chain {i+1}: n_samples={len(lp)}  accept={ar:.3f}  "
              f"logp mean={finite.mean():.1f}  std={finite.std():.1f}")

    # --- R-hat ---
    rhat = gelman_rubin(msgr_samples)
    rhat_cl = rhat[:N_LNCL]
    rhat_alm = rhat[N_LNCL:]
    print(f"\nR-hat (ln Cl):  max={rhat_cl.max():.4f}  median={np.median(rhat_cl):.4f}")
    print(f"R-hat (alm):    max={rhat_alm.max():.4f}  median={np.median(rhat_alm):.4f}")
    print(f"Parameters with R-hat < 1.1: {(rhat < 1.1).mean() * 100:.1f}%")

    # --- ESS (ln Cl block; this is what Phase 0's table reports) ---
    try:
        ess = compute_ess(msgr_samples)
        ess_cl = ess[:N_LNCL]
        n_post_burnin = min(c.shape[0] for c in msgr_samples)
        print(f"\nESS (ln Cl), out of {n_post_burnin} post-burn-in samples/chain "
              f"x {n_chains} chains = {n_post_burnin * n_chains} total:")
        print(f"  median={np.median(ess_cl):.1f}  min={ess_cl.min():.1f}  max={ess_cl.max():.1f}")
        print(f"  efficiency (median ESS / total draws) = "
              f"{np.median(ess_cl) / (n_post_burnin * n_chains) * 100:.1f}%")
    except Exception as e:
        print(f"\nWarning: ESS computation failed: {e}")
        ess_cl = None

    # --- logp trace plot ---
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, lp in enumerate(msgr_logp):
        ax.plot(lp, label=f"Chain {i+1}", alpha=0.75, lw=0.8)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Log-posterior")
    ax.set_title(f"Messenger sampler log-posterior traces (L={LMAX})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"traces_messenger_L{LMAX}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved trace plot -> {path}")

    # --- ESS histogram ---
    if ess_cl is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(ess_cl, bins=50, color="steelblue", edgecolor="none")
        n_total = n_post_burnin * n_chains
        ax.axvline(n_total, color="red", ls="--", label=f"N={n_total} (ideal, IAT=1)")
        ax.set_xlabel("ESS (ln Cl)")
        ax.set_ylabel("Count")
        ax.set_title(f"Messenger sampler ESS per ln(Cl) coefficient (L={LMAX})")
        ax.legend()
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"ess_messenger_L{LMAX}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved ESS histogram -> {path}")

    # --- C_l agreement with the Phase 0 float64 HMC reference ---
    ref_samples, ref_logp, ref_accept = load_cmb_chains(REFERENCE_DIR)
    if len(ref_samples) == 0:
        print(f"\nReference dir not found or empty: {REFERENCE_DIR} -- skipping C_l comparison")
    else:
        msgr_all = np.concatenate(msgr_samples, axis=0)
        ref_all = np.concatenate(ref_samples, axis=0)
        ells = np.arange(2, LMAX)

        cl_msgr = np.exp(msgr_all[:, :N_LNCL])
        cl_ref = np.exp(ref_all[:, :N_LNCL])

        cl_msgr_mean, cl_msgr_std = cl_msgr.mean(axis=0), cl_msgr.std(axis=0)
        cl_ref_mean, cl_ref_std = cl_ref.mean(axis=0), cl_ref.std(axis=0)

        # z-score of the mean difference, in combined-posterior-std units
        combined_std = np.sqrt(cl_msgr_std ** 2 / len(cl_msgr) + cl_ref_std ** 2 / len(cl_ref))
        z = (cl_msgr_mean - cl_ref_mean) / (combined_std + 1e-300)

        low_l_mask = ells <= 100
        print(f"\nC_l agreement vs Phase 0 float64 HMC reference ({REFERENCE_DIR}):")
        print(f"  l<=100: mean|z|={np.abs(z[low_l_mask]).mean():.2f}  max|z|={np.abs(z[low_l_mask]).max():.2f}")
        print(f"  all l:  mean|z|={np.abs(z).mean():.2f}  max|z|={np.abs(z).max():.2f}")

        try:
            from diffcmb.power import call_CAMB_map
            cl_lcdm = call_CAMB_map(LCDM_PARAMS, LMAX)
            have_lcdm = True
        except Exception:
            have_lcdm = False

        dl_msgr = ells * (ells + 1) * cl_msgr_mean / (2 * np.pi)
        dl_ref = ells * (ells + 1) * cl_ref_mean / (2 * np.pi)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
        axes[0].plot(ells, dl_msgr, color="steelblue", lw=1.5, label="Messenger (posterior mean)")
        axes[0].plot(ells, dl_ref, color="coral", lw=1.2, ls="--", label="Phase 0 HMC (posterior mean)")
        if have_lcdm:
            dl_lcdm = ells * (ells + 1) * cl_lcdm[2:LMAX] / (2 * np.pi)
            axes[0].plot(ells, dl_lcdm, "k:", lw=1, label="ΛCDM fiducial")
        axes[0].set_ylabel(r"$D_\ell = \ell(\ell+1)C_\ell / 2\pi$")
        axes[0].set_title(f"Messenger vs Phase 0 HMC power spectrum (L={LMAX})")
        axes[0].legend()

        axes[1].plot(ells, z, color="steelblue", lw=1)
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].axhline(3, color="red", ls="--", lw=0.8)
        axes[1].axhline(-3, color="red", ls="--", lw=0.8)
        axes[1].set_xlabel(r"$\ell$")
        axes[1].set_ylabel("z (msgr - ref)")
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"messenger_vs_hmc_L{LMAX}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved comparison plot -> {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
