import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.cmb import load_cmb_chains

LCDM_PARAMS = [67.74, 0.0486, 0.2589, 0.06, 0.0, 0.066]
OUT_DIR = "results/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

def gelman_rubin(chains):
    M = len(chains)
    N = min(c.shape[0] for c in chains)
    chains = np.stack([c[:N] for c in chains], axis=0)   # (M, N, P)
    chain_means = chains.mean(axis=1)                     # (M, P)
    grand_mean  = chain_means.mean(axis=0)                # (P,)
    B = N / (M - 1) * ((chain_means - grand_mean) ** 2).sum(axis=0)
    W = (((chains - chain_means[:, None, :]) ** 2).sum(axis=1) / (N - 1)).mean(axis=0)
    var_hat = (N - 1) / N * W + B / N
    return np.sqrt(var_hat / (W + 1e-30))

def compute_ess(chains):
    try:
        import tensorflow as tf
        import tensorflow_probability as tfp
        stacked = np.stack(chains, axis=1) # (N, M, P)
        ess_tf = tfp.mcmc.effective_sample_size(stacked)
        return ess_tf.numpy()
    except Exception as e:
        print(f"Warning: Could not compute ESS using TFP: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate MCMC Comparison Dashboard.")
    parser.add_argument("--lmax", type=int, default=200)
    parser.add_argument("--nside", type=int, default=128)
    args = parser.parse_args()

    lmax = args.lmax
    nside = args.nside
    n_lncl = lmax - 2

    print("=== MCMC Comparative Dashboard Generation ===")
    print(f"Target: lmax={lmax}, nside={nside}\n")

    samplers = {
        "Preconditioned HMC": {
            "dir": f"results/lmax{lmax}_nside{nside}_hmc_real_preconditioned",
            "color": "steelblue",
            "alpha": 0.3,
            "label": "Preconditioned HMC"
        },
        "Gibbs (Frozen Nuisance)": {
            "dir": f"results/lmax{lmax}_nside{nside}_gibbs_real_frozen",
            "color": "coral",
            "alpha": 0.3,
            "label": "Gibbs (Frozen Nuisance)"
        },
        "Gibbs (Deep MAP Stabilized)": {
            "dir": f"results/lmax{lmax}_nside{nside}_gibbs_real",
            "color": "forestgreen",
            "alpha": 0.4,
            "label": "Gibbs (Deep MAP)"
        }
    }

    # Load fiducial Cl
    try:
        from src.cmb.power import call_CAMB_map
        cl_lcdm = call_CAMB_map(LCDM_PARAMS, lmax)
        have_lcdm = True
        ells = np.arange(2, lmax)
        dl_lcdm = ells * (ells + 1) * cl_lcdm[2:lmax] / (2 * np.pi)
    except Exception:
        have_lcdm = False

    # Setup plots
    fig_ps, ax_ps = plt.subplots(figsize=(12, 7))
    if have_lcdm:
        ax_ps.plot(ells, dl_lcdm, "r--", lw=2.0, label="ΛCDM Fiducial Model")

    fig_rh, ax_rh = plt.subplots(figsize=(10, 5))

    report_md = []
    report_md.append(f"# MCMC Comparison Dashboard (lmax={lmax}, nside={nside})\n")
    report_md.append("| Sampler | Active Chains | Avg Accept Rate | Median $R$-hat ($\\ln C_\\ell$) | Conv. Fraction ($R$-hat < 1.1) | Median ESS ($\\ln C_\\ell$) |")
    report_md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    any_loaded = False

    for name, config in samplers.items():
        r_dir = config["dir"]
        if not os.path.exists(r_dir) or len(os.listdir(r_dir)) == 0:
            print(f"[{name}] Directory not found or empty: {r_dir}. Skipping.")
            continue

        any_loaded = True
        chains_samples, chains_logprob, chains_accepted = load_cmb_chains(r_dir)
        n_chains = len(chains_samples)

        # Calculate diagnostics
        avg_accept = np.mean(chains_accepted)
        rhat = gelman_rubin(chains_samples) if n_chains >= 2 else None
        ess = compute_ess(chains_samples) if n_chains >= 2 else None

        rhat_median_cl = np.median(rhat[:n_lncl]) if rhat is not None else np.nan
        rhat_conv_cl = (rhat[:n_lncl] < 1.1).mean() * 100 if rhat is not None else 0.0
        ess_median_cl = np.median(ess[:n_lncl]) if ess is not None else np.nan

        report_md.append(f"| {name} | {n_chains} | {avg_accept:.3f} | {rhat_median_cl:.4f} | {rhat_conv_cl:.1f}% | {ess_median_cl:.1f} |")

        # Plot power spectrum
        all_samples = np.concatenate(chains_samples, axis=0)
        cl_samps = np.exp(all_samples[:, :n_lncl])
        cl_mean = cl_samps.mean(axis=0)
        cl_lo = np.percentile(cl_samps, 16, axis=0)
        cl_hi = np.percentile(cl_samps, 84, axis=0)

        ells = np.arange(2, lmax)
        dl_mean = ells * (ells + 1) * cl_mean / (2 * np.pi)
        dl_lo = ells * (ells + 1) * cl_lo / (2 * np.pi)
        dl_hi = ells * (ells + 1) * cl_hi / (2 * np.pi)

        ax_ps.plot(ells, dl_mean, color=config["color"], lw=1.8, label=f"{config['label']} Mean")
        ax_ps.fill_between(ells, dl_lo, dl_hi, color=config["color"], alpha=config["alpha"], label=f"{config['label']} 68% CI")

        # Plot R-hat histogram
        if rhat is not None:
            ax_rh.hist(rhat[:n_lncl], bins=50, alpha=0.5, color=config["color"], label=config["label"], density=True)

    if not any_loaded:
        print("Error: No sampler results could be loaded!")
        sys.exit(1)

    # Finalize plots
    ax_ps.set_xlabel(r"$\ell$", fontsize=12)
    ax_ps.set_ylabel(r"$D_\ell = \ell(\ell+1)C_\ell / 2\pi$ ($\mu$K$^2$)", fontsize=12)
    ax_ps.set_title(f"Reconstructed CMB Power Spectrum Comparison (lmax={lmax})", fontsize=14)
    ax_ps.legend(fontsize=10, loc="upper right")
    ax_ps.grid(True, alpha=0.3)
    ps_path = os.path.join(OUT_DIR, "power_spectrum_comparison.png")
    fig_ps.savefig(ps_path, dpi=150)
    plt.close(fig_ps)
    print(f"Saved power spectrum comparison plot -> {ps_path}")

    ax_rh.axvline(1.1, color="red", ls="--", lw=1.5, label="R-hat = 1.1 Limit")
    ax_rh.set_xlabel("R-hat", fontsize=12)
    ax_rh.set_ylabel("Density", fontsize=12)
    ax_rh.set_title("Gelman-Rubin R-hat Distribution Comparison", fontsize=14)
    ax_rh.legend(fontsize=10)
    ax_rh.grid(True, alpha=0.3)
    rh_path = os.path.join(OUT_DIR, "rhat_comparison.png")
    fig_rh.savefig(rh_path, dpi=150)
    plt.close(fig_rh)
    print(f"Saved R-hat comparison plot -> {rh_path}")

    # Write dashboard report
    report_path = os.path.join(OUT_DIR, "dashboard.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_md))
        f.write("\n\n## Comparison Visualizations\n")
        f.write("### 1. Angular Power Spectrum $D_\\ell$\n")
        f.write("![CMB Power Spectrum Comparison](power_spectrum_comparison.png)\n\n")
        f.write("### 2. Gelman-Rubin $R$-hat Convergence Distribution\n")
        f.write("![R-hat Comparison](rhat_comparison.png)\n")

    print(f"Saved dashboard report -> {report_path}")

if __name__ == "__main__":
    main()
