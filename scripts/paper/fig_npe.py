"""Learned vs exact posterior (ROADMAP T1.1): the benchmark's figure.

Reads the saved products of one flow directory, never the flow itself:
  vs_exact_strict.npz  npe_vs_exact.py, 48 strict skies (shift, width ratio, correlations)
  sbc_heldout.npz      npe_sbc.py, held-out SBC ranks per bin
  c2st_strict.npz      npe_c2st.py, per-sky classifier accuracies (optional)

Panels (one PDF each, single column):
  npe_width_ratio   learned sd / exact sd per bin, mean over skies with per-sky points
                    (the headline: the information the spectra-level summaries lose);
  npe_width_ratio_ell
                    the same per multipole, mean +- SE over skies (only if the product has
                    width_ratio_ell, saved by npe_vs_exact.py since 2026-10-21): does the gap
                    sit at every l or a few? The band edge is shaded;
  npe_shift         (learned mean - exact mean) / exact sd per bin, mean +- SE over skies;
  npe_sbc_cltt, npe_sbc_clpp
                    held-out SBC rank histograms per bin against the simulated
                    discrete-uniform band (fig1_validation.uniform_band);
  npe_corr          mean correlation matrix of the 8 bin-averaged ln C, learned - exact;
  npe_c2st          per-sky C2ST accuracy: raw, shape-only (each class standardised),
                    and the exact-vs-exact nulls.
The [60,64) band edge is drawn hollow and left out of every pooled number
(npe_vs_exact.py's convention). Caption facts are printed, not drawn.

Usage:
  PYTHONPATH=diffcmb:scripts:scripts/paper .venv/bin/python scripts/paper/fig_npe.py \
      --flow results/analysis/npe_sims_l64_A3000_n30_nu30/flow_maf5x256 --outdir <plots>/appendix
      (make figure-npe then moves the three headline panels to <plots>/figure7)
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps  # noqa: E402

ps.apply()  # installs Agg + paper rcParams before pyplot is touched
import matplotlib.pyplot as plt  # noqa: E402
from fig1_validation import N_HIST_BINS, uniform_band  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

BAND_EDGE_LO = 60  # npe_vs_exact.BAND_EDGE_LO: reported, never pooled
HEADLINE_BIN = (30, 60)
# npe_sbc.bin_slices block names -> (label, colour, x offset so the two spectra do not overlap)
TT, PP = "C_l^TT", "C_L^phiphi"
BLOCKS = {TT: (r"$C_\ell^{TT}$", ps.COL_CLTT, -0.8), PP: (r"$C_L^{\phi\phi}$", ps.COL_CLPP, 0.8)}


def load(flow_dir):
    """The three products, checked to come from this flow and to share one bin grid."""
    vs = np.load(os.path.join(flow_dir, "vs_exact_strict.npz"))
    sbc = np.load(os.path.join(flow_dir, "sbc_heldout.npz"))
    c2st_path = os.path.join(flow_dir, "c2st_strict.npz")
    c2st = np.load(c2st_path) if os.path.exists(c2st_path) else None
    flow = os.path.normpath(flow_dir)
    for name, d in (("vs_exact_strict", vs), ("sbc_heldout", sbc), ("c2st_strict", c2st)):
        if d is not None and os.path.normpath(str(d["flow"])) != flow:
            raise SystemExit(f"{name}.npz was made by {d['flow']}, not {flow_dir}")
    if not (np.array_equal(vs["bins"], sbc["bins"])
            and np.array_equal(vs["bin_blocks"], sbc["bin_blocks"])):
        raise SystemExit("vs_exact_strict and sbc_heldout use different bins")
    return vs, sbc, c2st


def _bin_label(lo, hi):
    return f"[{lo},{hi})"


def _centres(bins):
    return 0.5 * (bins[:, 0] + bins[:, 1] - 1)


def _per_block(ax, vs, values, err, ylabel, ref):
    """Mean +- err per bin for both spectra; the band edge hollow."""
    bins, blocks = vs["bins"], vs["bin_blocks"]
    for block, (label, col, off) in BLOCKS.items():
        sel = blocks == block
        x = _centres(bins[sel]) + off
        edge = bins[sel][:, 0] >= BAND_EDGE_LO
        for mask, face in ((~edge, col), (edge, "white")):
            ax.errorbar(x[mask], values[sel][mask], yerr=err[sel][mask], fmt="o", color=col,
                        mfc=face, ms=4, capsize=2, lw=1.0, elinewidth=0.8,
                        label=label if face == col else None)
    ax.axhline(ref, color="0.45", lw=0.7, ls="--", zorder=0)
    ax.set_xlim(0, 64)
    ax.set_xlabel(r"multipole $\ell$ or $L$ (bin centre)")
    ax.set_ylabel(ylabel)


def _headline(vs, block):
    j = np.flatnonzero((vs["bin_blocks"] == block) & (vs["bins"][:, 0] == HEADLINE_BIN[0])
                       & (vs["bins"][:, 1] == HEADLINE_BIN[1]))[0]
    return j


def plot_width_ratio(vs, outpath):
    wr = vs["width_ratio"]
    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    rng = np.random.default_rng(3)
    for block, (_, col, off) in BLOCKS.items():
        sel = np.flatnonzero(vs["bin_blocks"] == block)
        for j in sel:
            x = _centres(vs["bins"][j:j + 1])[0] + off + rng.uniform(-0.35, 0.35, wr.shape[0])
            ax.scatter(x, wr[:, j], s=2, color=col, alpha=0.25, lw=0, zorder=1)
    se = wr.std(0, ddof=1) / np.sqrt(wr.shape[0])
    _per_block(ax, vs, wr.mean(0), se, "width, learned / exact", 1.0)
    ax.legend(loc="upper left", handletextpad=0.3)
    jt, jp = _headline(vs, TT), _headline(vs, PP)
    ps.stat_box(ax, f"{_bin_label(*HEADLINE_BIN)}: {wr[:, jt].mean():.2f} / {wr[:, jp].mean():.2f}",
                loc="lower right")
    ps.save(fig, outpath)


def per_ell_mean(vs, lo, hi):
    """Mean over skies and over l in [lo, hi) of the per-multipole width ratio, per block."""
    w = vs["width_ratio_ell"].mean(0)
    n = w.size // 2
    sel = slice(lo - 2, hi - 2)
    return {TT: w[:n][sel].mean(), PP: w[n:][sel].mean()}


def plot_width_ratio_ell(vs, outpath):
    w = vs["width_ratio_ell"]
    n = w.shape[1] // 2
    ell = np.arange(2, 2 + n)
    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    ax.axvspan(BAND_EDGE_LO - 0.5, ell[-1] + 0.5, color="0.92", lw=0, zorder=0)
    for k, (label, col, _) in enumerate(BLOCKS.values()):  # TT columns first, then PP
        x = w[:, k * n:(k + 1) * n]
        m, se = x.mean(0), x.std(0, ddof=1) / np.sqrt(x.shape[0])
        ax.fill_between(ell, m - se, m + se, color=col, alpha=0.25, lw=0)
        ax.plot(ell, m, color=col, lw=1.0, label=label)
    ax.axhline(1.0, color="0.45", lw=0.7, ls="--", zorder=0)
    ax.set_xlim(1.5, ell[-1] + 0.5)
    ax.set_ylim(min(0.85, w.mean(0).min() - 0.05), None)  # room for the stat box below the 1.0 line
    ax.set_xlabel(r"multipole $\ell$ or $L$")
    ax.set_ylabel("width, learned / exact")
    ax.legend(loc="upper left", handletextpad=0.3)
    h = per_ell_mean(vs, *HEADLINE_BIN)
    ps.stat_box(ax, f"{_bin_label(*HEADLINE_BIN)}: {h[TT]:.2f} / {h[PP]:.2f}", loc="lower right")
    ps.save(fig, outpath)


def plot_shift(vs, outpath):
    sh = vs["shift"]
    n = sh.shape[0]
    mean, se = sh.mean(0), sh.std(0, ddof=1) / np.sqrt(n)
    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    _per_block(ax, vs, mean, se, "shift, (learned $-$ exact) / exact sd", 0.0)
    ax.legend(loc="upper left", handletextpad=0.3)
    pooled = vs["bins"][:, 0] < BAND_EDGE_LO
    zmax = np.max(np.abs(mean[pooled] / se[pooled]))
    ps.stat_box(ax, rf"max $|z|={zmax:.1f}$, $N={n}$", loc="lower right")
    ps.save(fig, outpath)
    return mean / se


def plot_sbc(sbc, block, outpath):
    n_draws = int(sbc["n_draws"])
    sel = np.flatnonzero(sbc["bin_blocks"] == block)
    ranks = sbc["bin_ranks"]
    label = BLOCKS[block][0]
    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    b68, b95 = uniform_band(ranks.shape[0], n_draws)
    edges = np.linspace(0, 1, N_HIST_BINS + 1)
    ax.stairs(b95[1], edges, baseline=b95[0], fill=True, color=ps.COL_NULL, alpha=0.45, lw=0,
              label="uniform, 95%")
    ax.stairs(b68[1], edges, baseline=b68[0], fill=True, color=ps.COL_NULL, alpha=0.75, lw=0,
              label="uniform, 68%")
    shades = plt.get_cmap("viridis")(np.linspace(0.0, 0.85, sel.size))
    for c, j in zip(shades, sel):
        u = (ranks[:, j] + 0.5) / (n_draws + 1.0)
        lo, hi = sbc["bins"][j]
        ax.hist(u, bins=N_HIST_BINS, range=(0, 1), histtype="step", density=True, color=c,
                lw=1.1, ls=":" if lo >= BAND_EDGE_LO else "-", label=_bin_label(lo, hi))
    ax.axhline(1.0, color="0.45", lw=0.7, ls="--", zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.7, 1.45)  # N = 2000: the 95% band is +-0.13, so a zero baseline hides it
    ax.set_xlabel(f"normalised rank $u$, {label}")
    ax.set_ylabel("density")
    ax.legend(loc="upper center", ncol=3, fontsize=6, columnspacing=0.8, handlelength=1.4)
    ps.stat_box(ax, rf"min $p={sbc['bin_p'][sel].min():.2f}$, $N={ranks.shape[0]}$",
                loc="lower left")
    ps.save(fig, outpath)


def plot_corr(vs, outpath):
    diff = vs["corr_learned"].mean(0) - vs["corr_exact"].mean(0)
    short = {TT: "T", PP: r"\phi"}
    names = [short[b] + _bin_label(lo, hi) for b, (lo, hi) in zip(vs["bin_blocks"], vs["bins"])]
    lim = 0.2
    fig, ax = plt.subplots(figsize=(ps.FIG_1COL[0], ps.FIG_1COL[0] * 0.88))
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if i != j:
                ax.text(j, i, f"{diff[i, j]:+.2f}".replace("+0.", "+.").replace("-0.", "-."),
                        ha="center", va="center", fontsize=5)
    ticks = [f"${n}$" for n in names]
    ax.set_xticks(range(len(names)), ticks, rotation=90, fontsize=6)
    ax.set_yticks(range(len(names)), ticks, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=r"$r_{\rm learned}-r_{\rm exact}$")
    off = ~np.eye(diff.shape[0], dtype=bool)
    ps.save(fig, outpath)
    return np.max(np.abs(diff[off]))


def plot_c2st(c2st, outpath):
    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    edges = np.linspace(0.3, 1.0, 29)
    series = (("raw", "learned vs exact", ps.COL_AWARE, "-"),
              ("null", "exact vs exact", ps.COL_AWARE, ":"),
              ("shape", "learned vs exact, standardised", ps.COL_BLIND, "-"),
              ("null_shape", "exact vs exact, standardised", ps.COL_BLIND, ":"))
    for key, label, col, ls in series:
        ax.hist(c2st[key], bins=edges, histtype="step", color=col, ls=ls, lw=1.2, label=label)
    ax.axvline(0.5, color="0.45", lw=0.7, ls="--", zorder=0)
    ax.set_xlabel("classifier accuracy per sky")
    ax.set_ylabel("skies")
    ax.set_xlim(edges[0], edges[-1])
    ax.margins(y=0.35)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper right", fontsize=6, handlelength=1.6)
    ps.stat_box(ax, f"raw {c2st['raw'].mean():.3f}\nshape {c2st['shape'].mean():.3f} "
                f"(null {c2st['null_shape'].mean():.3f})", loc="upper left")
    ps.save(fig, outpath)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--flow", required=True, help="flow directory holding the saved products")
    p.add_argument("--outdir", required=True)
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    vs, sbc, c2st = load(a.flow)
    out = lambda name: os.path.join(a.outdir, f"{name}.pdf")  # noqa: E731

    plot_width_ratio(vs, out("npe_width_ratio"))
    has_ell = "width_ratio_ell" in vs.files
    if has_ell:
        plot_width_ratio_ell(vs, out("npe_width_ratio_ell"))
    z = plot_shift(vs, out("npe_shift"))
    for block, name in ((TT, "npe_sbc_cltt"), (PP, "npe_sbc_clpp")):
        plot_sbc(sbc, block, out(name))
    dmax = plot_corr(vs, out("npe_corr"))
    if c2st is not None:
        plot_c2st(c2st, out("npe_c2st"))

    wr = vs["width_ratio"]
    print("\nCAPTION FACTS (do not draw these into the panels):")
    print(f"  flow {a.flow}; exact reference {vs['chains_dir']} ({wr.shape[0]} strict skies, "
          f"{int(vs['n_learned'])} learned draws each)")
    print(f"  held-out SBC: {sbc['bin_ranks'].shape[0]} sims x {int(sbc['n_draws'])} draws; "
          f"per-bin p {np.array2string(sbc['bin_p'], precision=2)}")
    print(f"  {'block':>10} {'bin':>8} {'width':>6} {'shift':>7} {'z':>6}")
    for j, (b, (lo, hi)) in enumerate(zip(vs["bin_blocks"], vs["bins"])):
        flag = "  band edge, not pooled" if lo >= BAND_EDGE_LO else ""
        print(f"  {b:>10} {_bin_label(lo, hi):>8} {wr[:, j].mean():6.2f} "
              f"{vs['shift'][:, j].mean():+7.2f} {z[j]:+6.2f}{flag}")
    print(f"  correlation structure: max |r_learned - r_exact| = {dmax:.3f}")
    if has_ell:
        for lo, hi in ((2, 10), (10, 30), (30, 60), (60, 64)):
            h = per_ell_mean(vs, lo, hi)
            print(f"  width per multipole, {_bin_label(lo, hi)} mean: C_l^TT {h[TT]:.2f}, "
                  f"C_L^phiphi {h[PP]:.2f}")
    if c2st is not None:
        print(f"  C2ST mean accuracy: raw {c2st['raw'].mean():.3f}, shape {c2st['shape'].mean():.3f};"
              f" exact-vs-exact null raw {c2st['null'].mean():.3f}, "
              f"shape {c2st['null_shape'].mean():.3f}")


if __name__ == "__main__":
    main()
