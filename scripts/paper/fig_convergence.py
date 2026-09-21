"""Paper figure -- MCMC Convergence and Sampler Dynamics.

Produces three standalone vector PDFs:
  (a) convergence_trace_logp.pdf      -- Log-posterior traces across independent chains
  (b) convergence_rhat.pdf            -- Split-R_hat vs multipole for C_l^TT and C_L^phiphi
  (c) convergence_autocorr_ess.pdf    -- Autocorrelation time tau_int and ESS vs multipole

Adheres to paper_style.py:
- Standalone vector PDFs
- Okabe-Ito colors (COL_ALM, COL_PHI, COL_CLPP, COL_AWARE)
- Stat box with quantitative summary (e.g. median R_hat, ESS)
- No prose in panels

Usage:
  PYTHONPATH=diffcmb:scripts:scripts/paper .venv/bin/python scripts/paper/fig_convergence.py \
      --indir results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_packingv2 \
      --outdir /cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure_convergence
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paper_style


def split_rhat(x):
    """Split-Rhat on a single chain (split in half, treat as two chains)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x) // 2
    if n < 2:
        return np.nan
    c1, c2 = x[:n], x[n:2 * n]
    m1, m2 = np.mean(c1), np.mean(c2)
    v1, v2 = np.var(c1, ddof=1), np.var(c2, ddof=1)
    W = 0.5 * (v1 + v2)
    if W <= 1e-30:
        return 1.0
    B = n * 0.5 * ((m1 - 0.5 * (m1 + m2))**2 + (m2 - 0.5 * (m1 + m2))**2)
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def integrated_autocorr_time(x, max_lag=150):
    """Compute integrated autocorrelation time using Sokal adaptive window."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    n = len(x)
    if n < 2 or np.all(x == 0):
        return 1.0
    # FFT autocorrelation
    f = np.fft.rfft(x, n=2 * n)
    r = np.fft.irfft(f * np.conj(f))[:n]
    if r[0] <= 1e-30:
        return 1.0
    r = r / r[0]

    max_lag = min(max_lag, n // 3)
    tau = 1.0
    for lag in range(1, max_lag):
        if r[lag] <= 0:
            break
        tau += 2.0 * r[lag]
    return max(1.0, tau)


def generate_convergence_figures(indir: str, outdir: str, burn_in: int = 50) -> None:
    os.makedirs(outdir, exist_ok=True)
    paper_style.apply()

    chain_files = sorted(glob.glob(os.path.join(indir, "chain_r*.npz")))
    chain_files = [f for f in chain_files if not f.endswith("_ckpt.npz")]
    if not chain_files:
        raise FileNotFoundError(f"No chain files found in {indir}")

    print(f"Loading {len(chain_files)} chains from {indir}...")
    logp_list = []
    cl_tt_list = []
    cl_pp_list = []

    for f in chain_files:
        d = np.load(f)
        lmax = int(d["lmax"])
        n_cl = lmax - 2
        logp_list.append(d["logp"])
        cl_tt_list.append(d["alm_samples"][:, :n_cl])
        cl_pp_list.append(d["cl_phiphi_samples"])

    logp_arr = np.array(logp_list)  # (N_chains, N_sweeps)
    cl_tt_arr = np.array(cl_tt_list)  # (N_chains, N_sweeps, n_cl)
    cl_pp_arr = np.array(cl_pp_list)  # (N_chains, N_sweeps, n_cl)

    n_chains, n_sweeps, n_cl = cl_tt_arr.shape
    ell_arr = np.arange(2, lmax)

    # 1. Panel (a): Log-posterior trace
    print("Building panel (a): convergence_trace_logp.pdf...")
    fig_trace, ax_trace = plt.subplots(figsize=paper_style.FIG_1COL)
    sweeps = np.arange(n_sweeps)
    for i in range(min(12, n_chains)):
        ax_trace.plot(sweeps, logp_arr[i], lw=0.6, alpha=0.65, color=paper_style.COL_AWARE)
    ax_trace.axvline(burn_in, color="0.4", ls="--", lw=0.8)
    ax_trace.set_xlabel("Gibbs Sweep")
    ax_trace.set_ylabel(r"$\log \mathcal{P}(\mathrm{latent} \mid d)$")
    ax_trace.set_xlim(0, n_sweeps)
    paper_style.stat_box(ax_trace, f"$N = {n_chains}$ chains\nBurn-in $= {burn_in}$", loc="lower right")
    paper_style.save(fig_trace, os.path.join(outdir, "convergence_trace_logp.pdf"))

    # 2. Panel (b): Split-Rhat vs multipole
    print("Building panel (b): convergence_rhat.pdf...")
    rhat_tt = np.zeros((n_chains, n_cl))
    rhat_pp = np.zeros((n_chains, n_cl))

    for i in range(n_chains):
        for j in range(n_cl):
            rhat_tt[i, j] = split_rhat(cl_tt_arr[i, burn_in:, j])
            rhat_pp[i, j] = split_rhat(cl_pp_arr[i, burn_in:, j])

    med_rhat_tt = np.median(rhat_tt, axis=0)
    med_rhat_pp = np.median(rhat_pp, axis=0)

    fig_rhat, ax_rhat = plt.subplots(figsize=paper_style.FIG_1COL)
    ax_rhat.plot(ell_arr, med_rhat_tt, label=r"$C_\ell^{TT}$", color=paper_style.COL_ALM, lw=1.2)
    ax_rhat.plot(ell_arr, med_rhat_pp, label=r"$C_L^{\phi\phi}$", color=paper_style.COL_CLPP, lw=1.2)
    ax_rhat.axhline(1.05, color="0.4", ls="--", lw=0.8, label=r"Threshold $1.05$")
    ax_rhat.set_xlabel(r"Multipole $\ell, L$")
    ax_rhat.set_ylabel(r"Split-$\hat{R}$ (median across chains)")
    ax_rhat.set_ylim(0.98, 1.12)
    ax_rhat.legend(loc="upper right", frameon=False, fontsize=6.5)
    overall_max_rhat = max(np.median(med_rhat_tt), np.median(med_rhat_pp))
    paper_style.stat_box(ax_rhat, f"Median $\\hat{{R}} = {overall_max_rhat:.3f}$", loc="upper left")
    paper_style.save(fig_rhat, os.path.join(outdir, "convergence_rhat.pdf"))

    # 3. Panel (c): Autocorrelation time and ESS
    print("Building panel (c): convergence_autocorr_ess.pdf...")
    tau_tt = np.zeros((n_chains, n_cl))
    tau_pp = np.zeros((n_chains, n_cl))

    for i in range(n_chains):
        for j in range(n_cl):
            tau_tt[i, j] = integrated_autocorr_time(cl_tt_arr[i, burn_in:, j])
            tau_pp[i, j] = integrated_autocorr_time(cl_pp_arr[i, burn_in:, j])

    med_tau_tt = np.median(tau_tt, axis=0)
    med_tau_pp = np.median(tau_pp, axis=0)
    post_sweeps = n_sweeps - burn_in
    med_ess_tt = np.median(post_sweeps / med_tau_tt)
    med_ess_pp = np.median(post_sweeps / med_tau_pp)

    fig_tau, ax_tau = plt.subplots(figsize=paper_style.FIG_1COL)
    ax_tau.plot(ell_arr, med_tau_tt, color=paper_style.COL_ALM, label=r"$C_\ell^{TT}$", lw=1.2)
    ax_tau.plot(ell_arr, med_tau_pp, color=paper_style.COL_CLPP, label=r"$C_L^{\phi\phi}$", lw=1.2)
    ax_tau.set_xlabel(r"Multipole $\ell, L$")
    ax_tau.set_ylabel(r"Autocorr Time $\tau_{\mathrm{int}}$ [sweeps]")
    ax_tau.legend(loc="upper right", frameon=False, fontsize=6.5)
    paper_style.stat_box(
        ax_tau,
        f"Median $\\tau = {np.median(med_tau_tt):.1f}$ (ESS $\\approx {med_ess_tt:.0f}$)\n"
        f"Median $\\tau = {np.median(med_tau_pp):.1f}$ (ESS $\\approx {med_ess_pp:.0f}$)",
        loc="upper left",
    )
    paper_style.save(fig_tau, os.path.join(outdir, "convergence_autocorr_ess.pdf"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indir",
        default="results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_packingv2",
        help="Directory holding production chain files.",
    )
    parser.add_argument(
        "--outdir",
        default="/cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure_convergence",
        help="Directory to write panel PDFs.",
    )
    parser.add_argument(
        "--burn_in",
        type=int,
        default=50,
        help="Number of initial sweeps to discard as burn-in.",
    )
    args = parser.parse_args()
    generate_convergence_figures(args.indir, args.outdir, args.burn_in)


if __name__ == "__main__":
    main()
