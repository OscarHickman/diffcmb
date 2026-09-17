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

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paper_style as ps  # noqa: E402

ps.apply()  # installs Agg + paper rcParams before pyplot is touched
import matplotlib.pyplot as plt  # noqa: E402
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
    """The strongest bin pair, with the null it fails to clear drawn in.

    The regression line alone reads as a detection to the eye even when the
    caption says otherwise, so the 95% permutation-null cone is drawn behind
    it: the reader sees the fitted slope sitting inside the range of slopes
    the null itself produces. That is the honest presentation of a capability
    claim whose correlation is currently consistent with zero.
    """
    fig, ax = plt.subplots(figsize=ps.FIG_1COL_TALL)
    i, j = np.unravel_index(np.argmax(np.abs(corr) / null_abs95), corr.shape)
    x = np.concatenate([standardise(cl)[:, i] for cl, _, _ in pairs])
    y = np.concatenate([standardise(pp)[:, j] for _, pp, _ in pairs])
    ax.scatter(x, y, s=4, alpha=0.35, edgecolor="none", color=ps.COL_AWARE,
               rasterized=True)

    xs = np.linspace(x.min(), x.max(), 2)
    # The null cone: slopes corresponding to +/- the 95% null |r|. With both
    # axes standardised, the slope of a correlation r is just r.
    r_null = null_abs95[i, j]
    ax.fill_between(xs, r_null * xs, -r_null * xs, color=ps.COL_NULL,
                    alpha=0.55, lw=0, zorder=1,
                    label=rf"95% null, $|r|<{r_null:.2f}$")
    fit = np.polyfit(x, y, 1)
    ax.plot(xs, np.polyval(fit, xs), color=ps.COL_BLIND, lw=1.4, zorder=3,
            label=rf"measured, $r={corr[i, j]:+.3f}$")

    ax.axhline(0, color="0.8", lw=0.5, zorder=0)
    ax.axvline(0, color="0.8", lw=0.5, zorder=0)
    cl_lo, cl_hi = CL_BINS[i]
    pp_lo, pp_hi = CLPP_BINS[j]
    ax.set_xlabel(rf"$C_\ell^{{TT}}$, $\ell \in [{cl_lo},{cl_hi})$ (standardised)")
    ax.set_ylabel(rf"$C_L^{{\phi\phi}}$, $L \in [{pp_lo},{pp_hi})$ (standardised)")
    ax.legend(loc="upper left")
    ps.stat_box(ax, "strongest bin pair" "\n" "consistent with zero",
                loc="lower right")
    fig.tight_layout()
    ps.save(fig, outpath)


def plot_heatmap(corr, null_abs95, outpath, caveat):
    """All bin pairs. COLOUR ENCODES r / (that cell's own 95% null), not r.

    Two earlier encodings were both misleading, in opposite directions. On a
    fixed +/-0.6 scale every cell rendered near-white and the panel looked like
    an absence of data rather than a measured null. Rescaled to the largest
    null amplitude, the same cells looked like strong correlations, which
    overstates a result where nothing clears its null.

    Normalising each cell by its OWN null makes the colour mean one thing
    everywhere: |value| = 1 is exactly the 95% significance boundary. A cell
    that does not reach full saturation has not cleared chance, which is the
    claim this panel exists to support, and the printed annotation still gives
    the raw r so nothing is hidden by the transformation. The nulls differ
    cell to cell (the bins carry different numbers of effective draws), which
    is precisely why a single shared scale cannot encode significance.
    """
    ratio = corr / null_abs95
    fig, ax = plt.subplots(figsize=ps.FIG_1COL_TALL)
    im = ax.imshow(ratio, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(CLPP_BINS)))
    ax.set_xticklabels([f"[{a},{b})" for a, b in CLPP_BINS])
    ax.set_yticks(range(len(CL_BINS)))
    ax.set_yticklabels([f"[{a},{b})" for a, b in CL_BINS])
    ax.set_xlabel(r"$C_L^{\phi\phi}$ bin")
    ax.set_ylabel(r"$C_\ell^{TT}$ bin")
    ax.tick_params(top=False, right=False)
    for i in range(len(CL_BINS)):
        for j in range(len(CLPP_BINS)):
            sig = "*" if abs(corr[i, j]) > null_abs95[i, j] else ""
            ax.text(j, i, f"{corr[i, j]:+.2f}{sig}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if abs(ratio[i, j]) > 0.7 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("$r$ / 95% null  (|1| = significant)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    ps.save(fig, outpath)


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
    # Printed, NOT drawn: this is caption material. paper_style forbids prose
    # inside panels (see its docstring for why -- a stale burnt-in sentence is
    # unreachable by any test).
    caveat = (
        f"Source: {os.path.basename(args.indir.rstrip('/'))}, {len(files)} chains, "
        f"thin={args.thin} ({n_eff} pooled draws), proper C_L^phiphi prior nu={nu:g}. "
        "Capability claim, not a detection: no competing method produces this "
        "object at all. * marks entries outside the 95% permutation null."
    )
    print("\n  CAPTION TEXT (paste into main.tex, do not draw on the panel):\n"
          f"    {caveat}\n")
    plot_scatter(pairs, corr, null_abs95,
                os.path.join(args.outdir, "joint_posterior_scatter.pdf"), caveat)
    plot_heatmap(corr, null_abs95,
                os.path.join(args.outdir, "joint_posterior_heatmap.pdf"), caveat)


if __name__ == "__main__":
    main()
