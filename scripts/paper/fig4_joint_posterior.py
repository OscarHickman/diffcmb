"""Paper figure 4 -- the capability claim: the joint (C_l^TT, C_L^phiphi) posterior.

No competing method (MUSE, QE, Commander, diffusion, Flinch, Almanac) produces
this object at all -- a joint Gibbs sampler over (alm, C_l, phi, C_L^phiphi)
has the full joint in hand, so the cross-covariance between the two spectra is
measurable rather than assumed. Framed honestly per ROADMAP S1: this is
currently a NULL (1/16 cells above the permutation null, ~0.8 expected by
chance), so the figure is presented as "what this sampler is capable of
producing," not as a correlation detection.

Two standalone panel PDFs, factored out of plot_joint_cl_clpp_posterior.py's
combined two-axes PNG so each stands alone in a LaTeX figure* with lettered
caption prose:

  (a) joint_posterior_scatter.pdf -- the single bin pair with the largest
                                      |corr|/null ratio, standardised
                                      within-posterior draws.
  (b) joint_posterior_heatmap.pdf -- the full C_l bin x C_L^phiphi bin
                                      correlation matrix, cells above their
                                      95% permutation null starred.

This script re-runs the SAME within-chain-standardised / chain-bootstrap /
permutation-null pipeline as plot_joint_cl_clpp_posterior.py (imported, not
reimplemented) -- it only changes how the result is rendered.

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/paper/fig4_joint_posterior.py \
      --indir results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_doffix \
      --thin 45 \
      --outdir /cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure4
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plot_joint_cl_clpp_posterior import (  # noqa: E402
    CL_BINS,
    CLPP_BINS,
    bootstrap_corr,
    load_pairs,
    permutation_null,
    pooled_corr,
    standardise,
)


def plot_scatter(pairs, corr, null_abs95, outpath, caveat):
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    i, j = np.unravel_index(np.argmax(np.abs(corr) / null_abs95), corr.shape)
    x = np.concatenate([standardise(cl)[:, i] for cl, _, _ in pairs])
    y = np.concatenate([standardise(pp)[:, j] for _, pp, _ in pairs])
    ax.scatter(x, y, s=14, alpha=0.45, edgecolor="none", color="#1f4e79")
    fit = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 2)
    detected = abs(corr[i, j]) > null_abs95[i, j]
    ax.plot(xs, np.polyval(fit, xs), color="#c00000", lw=1.6,
            label=(f"r = {corr[i, j]:+.3f}  "
                   f"(95% null $|r|$ < {null_abs95[i, j]:.2f})"))
    ax.axhline(0, color="0.7", lw=0.6)
    ax.axvline(0, color="0.7", lw=0.6)
    cl_lo, cl_hi = CL_BINS[i]
    pp_lo, pp_hi = CLPP_BINS[j]
    ax.set_xlabel(rf"$C_\ell^{{TT}}$, $\ell \in [{cl_lo},{cl_hi})$  "
                  "(standardised per chain)")
    ax.set_ylabel(rf"$C_L^{{\phi\phi}}$, $L \in [{pp_lo},{pp_hi})$  (standardised)")
    ax.set_title("Strongest bin pair"
                 + ("" if detected else " -- still within the null"), fontsize=10)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.text(0.5, -0.06, caveat, ha="center", va="top", fontsize=7.5,
             wrap=True, color="0.25")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"wrote {outpath}")


def plot_heatmap(corr, null_abs95, outpath, caveat):
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    ax.set_xticks(range(len(CLPP_BINS)))
    ax.set_xticklabels([f"[{a},{b})" for a, b in CLPP_BINS], fontsize=8)
    ax.set_yticks(range(len(CL_BINS)))
    ax.set_yticklabels([f"[{a},{b})" for a, b in CL_BINS], fontsize=8)
    ax.set_xlabel(r"$C_L^{\phi\phi}$ bin")
    ax.set_ylabel(r"$C_\ell^{TT}$ bin")
    ax.set_title("Within-posterior correlation, all bin pairs", fontsize=10)
    for i in range(len(CL_BINS)):
        for j in range(len(CLPP_BINS)):
            sig = "*" if abs(corr[i, j]) > null_abs95[i, j] else ""
            ax.text(j, i, f"{corr[i, j]:+.2f}{sig}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(corr[i, j]) > 0.35 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.text(0.5, -0.06, caveat, ha="center", va="top", fontsize=7.5,
             wrap=True, color="0.25")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"wrote {outpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indir", required=True)
    ap.add_argument("--thin", type=int, default=45,
                   help="re-derive from the run's own tau_int; do not carry over")
    ap.add_argument(
        "--outdir",
        default="/cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure4")
    ap.add_argument("--n_rep", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    files, pairs = load_pairs(args.indir, args.thin)
    nu = pairs[0][2]
    n_eff = sum(cl.shape[0] for cl, _, _ in pairs)
    print(f"=== joint (C_l^TT, C_L^phiphi) posterior: {len(files)} chains, "
          f"thin={args.thin}, {n_eff} pooled draws ===")

    corr = pooled_corr(pairs)
    _boot = bootstrap_corr(pairs, args.n_rep, rng)  # kept for parity/repro; not plotted here
    null = permutation_null(pairs, args.n_rep, rng)
    null_abs95 = np.percentile(np.abs(null), 95, axis=0)

    n_above = int(np.sum(np.abs(corr) > null_abs95))
    n_cells = corr.size
    expected = 0.05 * n_cells
    print(f"  {n_above}/{n_cells} cells above their null; {expected:.1f} expected "
          "by chance -- "
          + ("NO detected correlation." if n_above <= expected + 1 else
             "excess over chance; inspect the pattern."))

    os.makedirs(args.outdir, exist_ok=True)
    caveat = (
        f"Source: {os.path.basename(args.indir.rstrip('/'))}, {len(files)} chains, "
        f"thin={args.thin} ({n_eff} pooled draws), proper C_L^phiphi prior nu={nu:g}. "
        "Capability claim, not a detection: no competing method produces this "
        "object at all. * marks entries outside the 95% permutation null."
    )
    plot_scatter(pairs, corr, null_abs95,
                os.path.join(args.outdir, "joint_posterior_scatter.pdf"), caveat)
    plot_heatmap(corr, null_abs95,
                os.path.join(args.outdir, "joint_posterior_heatmap.pdf"), caveat)


if __name__ == "__main__":
    main()
