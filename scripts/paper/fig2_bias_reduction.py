"""Paper figure 2 -- the headline effect: C_l^TT lensing-bias reduction.

Per ROADMAP.md's S1 decision this figure carries the paper's *validation*
argument: the lensing-aware joint sampler removes a lensing bias that a
lensing-blind (Commander-style) Gibbs fit leaves in C_l^TT, the residual is
consistent with zero, and the effect grows with l. Equivalently, we recover
A_L = 1 without a template or a nuisance parameter.

THREE standalone panel PDFs. The three answer the three questions a referee
asks in order -- is it there, is it a fluke, does it survive at the scales I
care about:

  (a) cl_comparison_single.pdf   -- is it there? One sky (seed 0), per-l-bin
                                    fractional bias, blind vs aware, at
                                    lmax=128.
  (b) bias_reduction_all_seeds.pdf -- is it a fluke? The same measurement on 4
                                    independent skies, lmax=128. Aware beats
                                    blind 4/4; blind deficit deepens with l
                                    4/4 (job 11991514; 93.7% +/- 1.8%).
  (c) bias_reduction_lmax192.pdf -- does it survive? The same measurement at
                                    lmax=192 (job 12015488), where the blind
                                    deficit reaches -8% and the reduction is
                                    98.4%.

SCALES ARE NEVER MIXED WITHIN A PANEL. (a) and (b) are lmax=128; (c) is
lmax=192 and is drawn as its own panel rather than swapped into (a), because
the 4-sky replication only exists at lmax=128 and overlaying the two would
imply a replication at 192 that was not run. The caption must say this.

(c) IS A BIAS-REDUCTION RESULT ONLY, NOT AN EXACTNESS CLAIM. The lmax=192 phi
block failed its equilibration gate (job 11987444, lag-1 0.976), so no
exactness statement above lmax=64 is available; the residual smallness of the
aware curve in (c) is not evidence of exactness at that scale.

The reference in every panel is the expected posterior mean of a CORRECT
unlensed model, E[C_l] = S_l(alm_true)/(k_l - 4) -- not the fiducial and not
the realized spectrum. compare_cl_bias_reduction.py's docstring explains why
at length; briefly, the flat-prior inverse-Gamma conditional has mean
S_l/(k_l-4) rather than S_l/k_l, an l-dependent offset common to both chains
that buries the signal if the realized spectrum is used instead.

Unreliable (unequilibrated-phi) bins are omitted from the panels and their
exclusion is stated in the caption; the analysis script's printed table always
reports every bin. The excluded set is lmax-dependent -- [2,10) at lmax=128,
and [2,10) + [10,30) at lmax=192 -- so it is read from
compare_cl_bias_reduction.unreliable_for rather than hardcoded here, which is
the bug that script carried until 2026-09-17.

No new computation: this script re-renders validated .npz outputs as paper
panels. Regenerate the inputs with compare_cl_bias_reduction.py, not by hand.

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/paper/fig2_bias_reduction.py \
      --figdir results/analysis/figures \
      --outdir /cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure2
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paper_style as ps  # noqa: E402

ps.apply()  # installs the Agg backend + paper rcParams before pyplot is used
import matplotlib.pyplot as plt  # noqa: E402
from compare_cl_bias_reduction import unreliable_for  # noqa: E402


def _rows(bins, lmax):
    """Indices of the bins that are safe to plot at this lmax."""
    bad = unreliable_for(lmax)
    return [i for i, (lo, hi) in enumerate(bins) if (lo, hi) not in bad]


def _frac(d, key, rows):
    return np.array([d[key][i] / d["truth"][i] - 1.0 for i in rows]) * 100.0


def _frac_err(d, key, rows):
    return np.array([d[key][i] / d["truth"][i] for i in rows]) * 100.0


def plot_single_realization(seed_npz, lmax, outpath):
    d = np.load(seed_npz)
    bins = d["bins"]
    rows = _rows(bins, lmax)

    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    x = np.arange(len(rows))
    w = 0.38
    for key, lab, col, off in (
            ("blind_mean", "lensing-blind", ps.COL_BLIND, -w / 2),
            ("aware_mean", "lensing-aware", ps.COL_AWARE, +w / 2)):
        ax.bar(x + off, _frac(d, key, rows), w,
               yerr=_frac_err(d, key.replace("mean", "sem"), rows),
               label=lab, color=col, capsize=2,
               error_kw={"lw": 0.7, "capthick": 0.7})
    ax.axhline(0.0, color=ps.COL_TRUTH, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"[{bins[i][0]},{bins[i][1]})" for i in rows])
    ax.set_xlabel(r"$\ell$ bin")
    ax.set_ylabel(r"bias in $C_\ell^{TT}$ (%)")
    ax.legend(loc="lower left")
    ps.stat_box(ax, r"one sky, $\ell_{\max}$=128", loc="upper right")
    ps.save(fig, outpath)


def plot_all_seeds(seed_npzs, lmax, outpath):
    """Colour encodes METHOD, not sky.

    The message is 'every aware curve sits at zero, every blind curve peels
    away'; the identity of an individual sky carries no information and a
    per-sky colour scale spent the reader's attention on it (8 legend entries
    for a 2-category claim). Skies are now repeated thin lines in the method's
    colour, so replication reads as line density rather than as a legend.
    """
    runs = [np.load(f) for f in seed_npzs]
    bins = runs[0]["bins"]
    rows = _rows(bins, lmax)
    ell_mid = np.array([(bins[i][0] + bins[i][1]) / 2.0 for i in rows])

    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    for key, lab, col, mk in (("blind_mean", "lensing-blind", ps.COL_BLIND, "o"),
                              ("aware_mean", "lensing-aware", ps.COL_AWARE, "s")):
        for j, d in enumerate(runs):
            ax.plot(ell_mid, _frac(d, key, rows), mk + "-", color=col,
                    alpha=0.75, ms=2.4, lw=0.9,
                    label=lab if j == 0 else None)
    ax.axhline(0.0, color=ps.COL_TRUTH, lw=0.8)
    ax.set_xlabel(r"$\ell$ (bin midpoint)")
    ax.set_ylabel(r"bias in $C_\ell^{TT}$ (%)")
    ax.legend(loc="lower left")
    ps.stat_box(ax, f"{len(runs)} independent skies" "\n"
                r"$\ell_{\max}$=128", loc="upper right", xy=(0.97, 0.62))
    ps.save(fig, outpath)


def plot_lmax192(npz, outpath):
    d = np.load(npz)
    bins = d["bins"]
    rows = _rows(bins, 192)
    ell_mid = np.array([(bins[i][0] + bins[i][1]) / 2.0 for i in rows])

    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    for key, lab, col, mk in (("blind_mean", "lensing-blind", ps.COL_BLIND, "o"),
                              ("aware_mean", "lensing-aware", ps.COL_AWARE, "s")):
        ax.plot(ell_mid, _frac(d, key, rows), mk + "-", color=col, ms=2.8,
                label=lab)
    ax.axhline(0.0, color=ps.COL_TRUTH, lw=0.8)
    ax.set_xlabel(r"$\ell$ (bin midpoint)")
    ax.set_ylabel(r"bias in $C_\ell^{TT}$ (%)")
    ax.legend(loc="lower left")
    ps.stat_box(ax, "one sky, " r"$\ell_{\max}$=192" "\n"
                "bias reduction only", loc="upper right", xy=(0.97, 0.62))
    ps.save(fig, outpath)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figdir", default="results/analysis/figures")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--lmax", type=int, default=128,
                    help="lmax of the per-seed inputs for panels (a) and (b)")
    ap.add_argument("--lmax192_npz",
                    default="results/analysis/figures/cl_bias_reduction_lmax192.npz")
    ap.add_argument(
        "--outdir",
        default="/cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure2")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    seed_npzs = []
    for s in args.seeds:
        f = os.path.join(args.figdir, f"bias_seed{s}.npz")
        if os.path.exists(f):
            seed_npzs.append(f)
        else:
            print(f"  ! missing, skipped: {f}")
    if not seed_npzs:
        raise SystemExit("no per-seed bias files found")

    plot_single_realization(seed_npzs[0], args.lmax,
                            os.path.join(args.outdir, "cl_comparison_single.pdf"))
    plot_all_seeds(seed_npzs, args.lmax,
                   os.path.join(args.outdir, "bias_reduction_all_seeds.pdf"))

    if os.path.exists(args.lmax192_npz):
        plot_lmax192(args.lmax192_npz,
                     os.path.join(args.outdir, "bias_reduction_lmax192.pdf"))
    else:
        print(f"  ! missing, panel (c) skipped: {args.lmax192_npz}")

    if len(seed_npzs) < 4:
        print(f"\n  NOTE: only {len(seed_npzs)}/4 seeds present -- rerun panel (b) "
              "when the rest land.")


if __name__ == "__main__":
    main()
