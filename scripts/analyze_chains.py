"""
Analyse completed NUTS chains for both synthetic and real data runs.
Produces a text summary and plots saved to results/analysis/.
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import argparse

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.cmb import load_cmb_chains

OUT_DIR = "results/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

LCDM_PARAMS = [67.74, 0.0486, 0.2589, 0.06, 0.0, 0.066]


def gelman_rubin(chains):
    """R-hat per parameter across a list of (n_samples, n_params) arrays."""
    M = len(chains)
    N = min(c.shape[0] for c in chains)
    chains = np.stack([c[:N] for c in chains], axis=0)   # (M, N, P)
    chain_means = chains.mean(axis=1)                     # (M, P)
    grand_mean  = chain_means.mean(axis=0)                # (P,)
    B = N / (M - 1) * ((chain_means - grand_mean) ** 2).sum(axis=0)
    W = (((chains - chain_means[:, None, :]) ** 2).sum(axis=1) / (N - 1)).mean(axis=0)
    var_hat = (N - 1) / N * W + B / N
    return np.sqrt(var_hat / (W + 1e-30))


def load_and_summarise(label, results_dir, lmax):
    if not os.path.exists(results_dir):
        print(f"[{label}] Directory not found: {results_dir}")
        return None, None, None

    chains_samples, chains_logprob, chains_accepted = load_cmb_chains(results_dir)
    n = len(chains_samples)
    if n == 0:
        print(f"[{label}] No chains found in {results_dir}")
        return None, None, None

    print(f"\n=== {label.upper()} ===")
    print(f"  Chains loaded : {n}")
    for i, (ar, lp) in enumerate(zip(chains_accepted, chains_logprob)):
        finite = lp[np.isfinite(lp)]
        print(f"  Chain {i+1}: accept_rate={ar:.3f}  "
              f"logp mean={finite.mean():.1f}  std={finite.std():.1f}  "
              f"n_samples={len(lp)}")

    # Gelman-Rubin (need ≥2 chains)
    if n >= 2:
        rhat = gelman_rubin(chains_samples)
        n_lncl = lmax - 2
        rhat_cl   = rhat[:n_lncl]
        rhat_alm  = rhat[n_lncl:]
        print(f"  R-hat (ln Cl):  max={rhat_cl.max():.4f}  median={np.median(rhat_cl):.4f}")
        print(f"  R-hat (alm):    max={rhat_alm.max():.4f}  median={np.median(rhat_alm):.4f}")
        converged = (rhat < 1.1).mean() * 100
        print(f"  Parameters with R-hat < 1.1: {converged:.1f}%")
    else:
        rhat = None
        print("  Only 1 chain — skipping Gelman-Rubin.")

    return chains_samples, chains_logprob, chains_accepted


def plot_traces(label, chains_logprob):
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, lp in enumerate(chains_logprob):
        ax.plot(lp, label=f"Chain {i+1}", alpha=0.75, lw=0.8)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Log-posterior")
    ax.set_title(f"NUTS log-posterior traces — {label}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"traces_{label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved trace plot → {path}")


def plot_power_spectrum(label, chains_samples, lmax):
    # Try to import CAMB for fiducial spectrum
    try:
        from src.cmb.power import call_CAMB_map
        cl_lcdm = call_CAMB_map(LCDM_PARAMS, lmax)
        have_lcdm = True
    except Exception:
        have_lcdm = False

    all_samples = np.concatenate(chains_samples, axis=0)
    n_lncl = lmax - 2
    ln_cl  = all_samples[:, :n_lncl]
    cl_samps = np.exp(ln_cl)
    ells = np.arange(2, lmax)

    cl_mean   = cl_samps.mean(axis=0)
    cl_lo     = np.percentile(cl_samps, 16, axis=0)
    cl_hi     = np.percentile(cl_samps, 84, axis=0)
    dl_mean   = ells * (ells + 1) * cl_mean / (2 * np.pi)
    dl_lo     = ells * (ells + 1) * cl_lo   / (2 * np.pi)
    dl_hi     = ells * (ells + 1) * cl_hi   / (2 * np.pi)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(ells, dl_lo, dl_hi, alpha=0.3, color="steelblue", label="68% CI")
    ax.plot(ells, dl_mean, color="steelblue", lw=1.5, label="Posterior mean")
    if have_lcdm:
        dl_lcdm = ells * (ells + 1) * cl_lcdm[2:lmax] / (2 * np.pi)
        ax.plot(ells, dl_lcdm, "r--", lw=1.5, label="ΛCDM fiducial")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell = \ell(\ell+1)C_\ell / 2\pi$")
    ax.set_title(f"Inferred CMB power spectrum — {label}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"power_spectrum_{label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved power spectrum → {path}")


def plot_rhat_histogram(label, chains_samples):
    if len(chains_samples) < 2:
        return
    rhat = gelman_rubin(chains_samples)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rhat, bins=50, color="steelblue", edgecolor="none")
    ax.axvline(1.1, color="red", ls="--", label="R-hat = 1.1")
    ax.set_xlabel("R-hat")
    ax.set_ylabel("Count")
    ax.set_title(f"Gelman-Rubin R-hat — {label}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"rhat_{label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved R-hat histogram → {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyse MCMC chains for CMB sampling.")
    parser.add_argument("--lmax", type=int, default=100)
    parser.add_argument("--nside", type=int, default=64)
    args = parser.parse_args()

    lmax = args.lmax
    nside = args.nside

    print(f"Analysis of NUTS chains — lmax={lmax}, nside={nside}")
    print(f"Output directory: {OUT_DIR}\n")

    runs = {
        "synthetic": f"results/lmax{lmax}_nside{nside}_nuts_synthetic",
        "real":      f"results/lmax{lmax}_nside{nside}_nuts_real",
    }

    for label, results_dir in runs.items():
        if not os.path.exists(results_dir):
            continue
        chains_samples, chains_logprob, chains_accepted = load_and_summarise(label, results_dir, lmax)
        if chains_samples is None:
            continue
        plot_traces(label, chains_logprob)
        plot_power_spectrum(label, chains_samples, lmax)
        plot_rhat_histogram(label, chains_samples)

    print("\nDone.")


if __name__ == "__main__":
    main()
