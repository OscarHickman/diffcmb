"""Paper schematic -- generative model (left) and the four-block Gibbs sampler (right).

One vector PDF at figure* width: schematic_forward_model.pdf.

Every formula here is copied from the code it describes, not paraphrased;
check against it whenever the sampler changes:
  Block 1  model.py::sample_cl_given_alm     InvGamma(k_l/2 - 1, S_l/2), k_l = 2l+1
  Block 4  lensing.py::sample_cl_phiphi_given_phi (proper prior)
           InvGamma((k_L + nu)/2, (S_L + nu C_L^fid)/2)
  truth    coverage_ensemble_chain.py: C_L^phiphi,true ~ InvGamma(nu/2, nu C_L^fid/2),
           C_l^TT,true = fiducial (the sampler's C_l prior is flat)
  operator lensing.py::precompute_lensing: bilinear interpolation (hp.get_interp_weights)
           at the deflected angles on the nside grid.
The first version of this figure had a stray "-1" in Block 4's shape, called
Block 2 "MCLMC / HMC" (production is HMC; MCLMC failed its stationarity gate),
and used d for both the deflection and the data. Panel letters live in the
caption, not the image.

Usage:
  PYTHONPATH=diffcmb:scripts/paper .venv/bin/python scripts/paper/fig_schematic.py
"""

from __future__ import annotations

import argparse
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import paper_style as ps

FS = 7.0          # body text in boxes: the journal's minimum legible size
FS_HEAD = 7.5
HEAD_FONT = "cmb10" if ps.JOURNAL in ("PRD", "JCAP") else None
GREY = "0.35"


def box(ax, x, y, w, h, head, body, color, alpha=0.10):
    for fc, a, z in ((color, alpha, 2), ("none", 1.0, 3)):
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.018",
            lw=0.9, edgecolor=color, facecolor=fc, alpha=a, zorder=z))
    ax.text(x + w / 2, y + h - 0.035, head, ha="center", va="top", fontsize=FS_HEAD,
            family=HEAD_FONT, color="0.1", zorder=4)
    ax.text(x + w / 2, y + (h - 0.07) / 2, body, ha="center", va="center",
            fontsize=FS, color="0.15", zorder=4, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, color=GREY, **kw):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9,
                                shrinkA=0, shrinkB=0, mutation_scale=8, **kw),
                zorder=3)


def panel_bg(ax, x, w, title):
    ax.add_patch(patches.FancyBboxPatch(
        (x, 0.0), w, 1.0, boxstyle="round,pad=0.0,rounding_size=0.02",
        lw=0.6, edgecolor="0.85", facecolor="0.985", zorder=1))
    ax.text(x + 0.015, 0.975, title, fontsize=8, family=HEAD_FONT, color="0.2",
            va="top", zorder=4)


def make(outdir: str) -> None:
    ps.apply()
    os.makedirs(outdir, exist_ok=True)
    fig = plt.figure(figsize=(ps.FIG_2COL[0], 3.55), layout="none")
    ax = fig.add_axes([0.004, 0.006, 0.992, 0.988])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # ---------------- left: generative model -------------------------------
    L0, LW = 0.0, 0.445
    panel_bg(ax, L0, LW, "Generative model")
    c1, c2, bw = 0.02, 0.232, 0.193
    box(ax, c1, 0.70, bw, 0.20, r"Primary spectrum",
        r"$C_\ell^{TT}$: fiducial $\Lambda$CDM" "\n" r"(flat prior in the sampler)",
        ps.COL_ALM)
    box(ax, c2, 0.70, bw, 0.20, r"Lensing spectrum",
        r"$C_L^{\phi\phi}\sim\mathrm{InvGamma}\!\left(\frac{\nu}{2},\frac{\nu C_L^{\rm fid}}{2}\right)$",
        ps.COL_CLPP)
    box(ax, c1, 0.42, bw, 0.22, r"Unlensed CMB",
        r"$a_{\ell m}\sim\mathcal{N}(0,C_\ell^{TT})$" "\n" r"$T(\hat n)=\sum a_{\ell m}Y_{\ell m}(\hat n)$",
        ps.COL_ALM)
    box(ax, c2, 0.42, bw, 0.22, r"Lensing potential",
        r"$\phi_{LM}\sim\mathcal{N}(0,C_L^{\phi\phi})$" "\n" r"deflection $\nabla\phi(\hat n)$",
        ps.COL_PHI)
    box(ax, 0.05, 0.215, 0.345, 0.15, r"Lensing operator",
        r"$\tilde T(\hat n)=T(\hat n+\nabla\phi)$, curved sky" "\n"
        r"(bilinear interpolation on HEALPix)", ps.COL_AWARE)
    box(ax, 0.05, 0.02, 0.345, 0.14, r"Data",
        r"$d(\hat n)=\tilde T(\hat n)+n(\hat n)$,  $n\sim\mathcal{N}(0,\mathbf{N})$", GREY,
        alpha=0.07)
    for cx in (c1 + bw / 2, c2 + bw / 2):
        arrow(ax, cx, 0.70, cx, 0.64)
    arrow(ax, c1 + bw / 2, 0.42, 0.17, 0.365)
    arrow(ax, c2 + bw / 2, 0.42, 0.275, 0.365)
    arrow(ax, 0.2225, 0.215, 0.2225, 0.16)

    # ---------------- right: Gibbs sampler ---------------------------------
    R0, RW = 0.46, 0.54
    panel_bg(ax, R0, RW, "Joint Gibbs sampler (one sweep)")
    bx, bwid, bh = 0.475, 0.455, 0.19
    ys = (0.715, 0.49, 0.265, 0.04)
    box(ax, bx, ys[0], bwid, bh, r"Block 1: $C_\ell^{TT}\,|\,a_{\ell m}$",
        r"$\mathrm{InvGamma}\!\left(\frac{2\ell+1}{2}-1,\,\frac{S_\ell}{2}\right)$,  "
        r"$S_\ell=\sum_m|a_{\ell m}|^2$" "\n" "exact conjugate draw", ps.COL_ALM)
    box(ax, bx, ys[1], bwid, bh, r"Block 2: $a_{\ell m}\,|\,C_\ell^{TT},\phi,d$",
        "Hamiltonian Monte Carlo; gradients by" "\n"
        "automatic differentiation through the SHT", ps.COL_ALM)
    box(ax, bx, ys[2], bwid, bh, r"Block 3: $\phi_{LM}\,|\,a_{\ell m},C_L^{\phi\phi},d$",
        "Hamiltonian Monte Carlo; gradients through" "\n"
        "the lensing operator (analytic adjoint)", ps.COL_PHI)
    box(ax, bx, ys[3], bwid, bh, r"Block 4: $C_L^{\phi\phi}\,|\,\phi_{LM}$",
        r"$\mathrm{InvGamma}\!\left(\frac{2L+1+\nu}{2},\,\frac{S^\phi_L+\nu C_L^{\rm fid}}{2}\right)$"
        "\n" r"exact conjugate draw, proper prior ($\nu=6$)", ps.COL_CLPP)
    xm = bx + bwid / 2
    for y_top, y_next in zip(ys[:-1], ys[1:]):
        arrow(ax, xm, y_top, xm, y_next + bh, color=ps.COL_AWARE)
    # return loop, Block 4 -> Block 1, drawn as an explicit path in the margin
    x0, xv = bx + bwid, 0.958
    y1, y4 = ys[0] + bh / 2, ys[3] + bh / 2
    ax.plot([x0, xv, xv], [y4, y4, y1], color=ps.COL_AWARE, lw=1.0, zorder=3,
            solid_joinstyle="round")
    arrow(ax, xv, y1, x0, y1, color=ps.COL_AWARE)
    ax.text(xv + 0.018, 0.5 * (y1 + y4), "next sweep", rotation=-90,
            ha="center", va="center", fontsize=FS, color=ps.COL_AWARE)

    ps.save(fig, os.path.join(outdir, "schematic_forward_model.pdf"))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir",
                   default="/cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure_schematic")
    make(p.parse_args().outdir)


if __name__ == "__main__":
    main()
