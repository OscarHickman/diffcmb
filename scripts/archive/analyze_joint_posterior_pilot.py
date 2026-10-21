"""
ROADMAP.md Section 1, Priority 2: the "one-chain question" sequencing fix
(2026-07-30) -- does the joint (C_l^TT, C_L^phiphi) posterior have a
correlation structure worth a figure at all, before funding the O(10-20)-chain
coverage ensemble?

Reuses the completed Priority-1 pilot chain (job 11663477,
results/analysis/pilot_coverage_lmax128_v2.npz -- lmax=128, phi_n_lfs=80,
Block 4 on) rather than a dedicated run, per the ROADMAP sequencing note: "one
chain with sample_cl_phiphi=True at any scale... no coverage ensemble, no new
code... They are the same job."

Caveat carried over honestly from the Priority-1 verdict: that pilot's own
equilibration gate returned NO-GO on lag-1 autocorrelation (worst 0.981,
achievements.md), though a deeper look showed decay-to-noise by lag~75-100 --
slow mixing, not a stuck chain. A non-fully-equilibrated chain can still show
whether a real cross-block correlation exists (correlation is a per-sweep joint
property, not something that requires full stationarity to appear at all), but
residual drift could inflate or fabricate apparent correlation if both blocks
drifted together across the burn-in/sampling boundary. This script reports
that risk explicitly rather than treating the number as final; job 11665429's
longer chain will be the follow-up check once it lands.
"""
import numpy as np

LOG_CL_BINS = [(2, 10), (10, 30), (30, 60), (60, 100), (100, 128)]


def bin_average(per_ell_values, ell_offset, bins):
    """Mean of per-ell values (indexed from `ell_offset`) within each bin."""
    n = per_ell_values.shape[-1]
    ell = np.arange(ell_offset, ell_offset + n)
    out = np.zeros((per_ell_values.shape[0], len(bins)))
    for j, (lo, hi) in enumerate(bins):
        mask = (ell >= lo) & (ell < hi)
        out[:, j] = per_ell_values[:, mask].mean(axis=1)
    return out


def main():
    path = "results/analysis/pilot_coverage_lmax128_v2.npz"
    d = np.load(path)
    lmax = int(d["lmax"])
    alm_samples = d["alm_samples"]
    ln_cl_phiphi = d["cl_phiphi_samples"]  # already log, see samplers.py

    n_lncl = lmax - 2
    ln_cl_tt = alm_samples[:, :n_lncl]  # log C_l^TT, l=2..lmax-1

    print(f"Loaded {ln_cl_tt.shape[0]} post-burn-in sweeps from {path}")
    print("Caveat: this pilot's equilibration gate verdict was NO-GO (lag-1 "
          "0.981); treat correlations below as suggestive, pending job "
          "11665429's longer-window confirmation.\n")

    bin_tt = bin_average(ln_cl_tt, 2, LOG_CL_BINS)
    bin_pp = bin_average(ln_cl_phiphi, 2, LOG_CL_BINS)

    n_bins = len(LOG_CL_BINS)
    corr = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            corr[i, j] = np.corrcoef(bin_tt[:, i], bin_pp[:, j])[0, 1]

    labels = [f"[{lo},{hi})" for lo, hi in LOG_CL_BINS]
    print("Pearson correlation: rows = ln C_l^TT bin, cols = ln C_L^phiphi bin")
    header = "            " + "  ".join(f"{lab:>9s}" for lab in labels)
    print(header)
    for i, lab in enumerate(labels):
        row = "  ".join(f"{corr[i, j]:9.3f}" for j in range(n_bins))
        print(f"  TT {lab:>7s}  {row}")

    worst = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    print(f"\nStrongest |correlation|: TT{labels[worst[0]]} vs "
          f"phiphi{labels[worst[1]]} = {corr[worst]:+.3f}")
    diag = np.array([corr[i, i] for i in range(n_bins)])
    print(f"Same-bin (diagonal) correlations: "
          f"{', '.join(f'{v:+.3f}' for v in diag)}")

    np.savez(
        "results/analysis/joint_posterior_pilot_correlation.npz",
        corr=corr, bins=np.array(LOG_CL_BINS), lmax=lmax,
        n_samples=ln_cl_tt.shape[0],
    )

    # --- Figure: correlation matrix heatmap ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n_bins))
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$C_L^{\phi\phi}$ bin")
    ax.set_ylabel(r"$C_\ell^{TT}$ bin")
    ax.set_title(
        f"Joint posterior correlation, ln-power (pilot, lmax={lmax},\n"
        f"n={ln_cl_tt.shape[0]} sweeps, equilibration NO-GO -- suggestive only)",
        fontsize=9,
    )
    for i in range(n_bins):
        for j in range(n_bins):
            v = corr[i, j]
            color = "white" if abs(v) > 0.6 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                     color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    out_png = "results/analysis/joint_posterior_pilot_correlation.png"
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved correlation matrix + figure to {out_png}")


if __name__ == "__main__":
    main()
