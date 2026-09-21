"""Paper schematic -- Forward model and 4-block differentiable Gibbs sampler.

Produces a publication-grade vector PDF:
  schematic_forward_model.pdf

Visualizes:
1. Physical generative forward model: C_l, C_L^phiphi -> a_lm, phi -> L(phi) -> d
2. 4-Block joint Gibbs sampler:
   - Block 1: C_l | a_lm ~ InvGamma (exact conjugate)
   - Block 2: a_lm | C_l, phi, d (gradient through SHT)
   - Block 3: phi | C_L^phiphi, a_lm, d (curved-sky HMC via AD)
   - Block 4: C_L^phiphi | phi ~ InvGamma (exact conjugate)

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/paper/fig_schematic.py \
      --outdir /cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure_schematic
"""

import argparse
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import paper_style


def draw_box(ax, x, y, w, h, text, color, title=None, alpha=0.15, text_color="0.1", fontsize=7.5, bold_title=True):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        linewidth=1.2,
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
        zorder=2
    )
    ax.add_patch(rect)
    # Border with full opacity
    rect_border = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        linewidth=1.2,
        edgecolor=color,
        facecolor="none",
        zorder=3
    )
    ax.add_patch(rect_border)

    cx = x + w / 2
    cy = y + h / 2
    if title:
        title_y = y + h - 0.08
        ax.text(cx, title_y, title, ha="center", va="top",
                fontsize=fontsize + 0.5, fontweight="bold" if bold_title else "normal",
                color=color, zorder=4)
        ax.text(cx, cy - 0.04, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, zorder=4, linespacing=1.2)
    else:
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, zorder=4, linespacing=1.2)


def draw_arrow(ax, x1, y1, x2, y2, color="0.4", width=1.2, head_width=0.025, head_length=0.03, zorder=3, style="-|>"):
    ax.annotate(
        "",
        xy=(x2, y2), xycoords="data",
        xytext=(x1, y1), textcoords="data",
        arrowprops={
            "arrowstyle": style,
            "color": color,
            "lw": width,
            "shrinkA": 2, "shrinkB": 2,
            "mutation_scale": 10,
        },
        zorder=zorder
    )


def make_schematic(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    paper_style.apply()

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Section 1: Forward Model (Left side: 0.03 to 0.45)
    # Background card for Forward Model
    bg_fwd = patches.FancyBboxPatch(
        (0.02, 0.04), 0.42, 0.92,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=0.8,
        edgecolor="0.8",
        facecolor="0.97",
        zorder=1
    )
    ax.add_patch(bg_fwd)
    ax.text(0.04, 0.92, "(a) Generative Forward Model", fontsize=8.5, fontweight="bold", color="0.2", zorder=4)

    # Node: Primordial Spectra
    draw_box(ax, 0.05, 0.72, 0.16, 0.14, r"$\mathcal{P}(C_\ell)$", paper_style.COL_ALM, title=r"$C_\ell^{TT}$ Prior")
    draw_box(ax, 0.25, 0.72, 0.16, 0.14, r"$\mathcal{P}(C_L^{\phi\phi})$", paper_style.COL_CLPP, title=r"$C_L^{\phi\phi}$ Prior")

    # Node: Latent Spherical Fields
    draw_box(ax, 0.05, 0.47, 0.16, 0.15, r"$a_{\ell m} \sim \mathcal{N}(0, C_\ell)$" + "\n" + r"$T(\hat{n}) = \mathrm{SHT}(a_{\ell m})$", paper_style.COL_ALM, title=r"Primary $T$")
    draw_box(ax, 0.25, 0.47, 0.16, 0.15, r"$\phi_{LM} \sim \mathcal{N}(0, C_L^{\phi\phi})$" + "\n" + r"$\mathbf{d}(\hat{n}) = \nabla \phi(\hat{n})$", paper_style.COL_PHI, title=r"Lensing $\phi$")

    draw_arrow(ax, 0.13, 0.72, 0.13, 0.62, color=paper_style.COL_ALM)
    draw_arrow(ax, 0.33, 0.72, 0.33, 0.62, color=paper_style.COL_CLPP)

    # Node: Curved-Sky Deflection & Lensing
    draw_box(ax, 0.10, 0.25, 0.26, 0.14, r"$\tilde{T}(\hat{n}) = T(\hat{n} + \nabla \phi(\hat{n}))$" + "\nHEALPix Deflection", paper_style.COL_AWARE, title=r"Lensing Operator $\mathcal{L}(\phi)$")

    draw_arrow(ax, 0.13, 0.47, 0.18, 0.39, color="0.4")
    draw_arrow(ax, 0.33, 0.47, 0.28, 0.39, color="0.4")

    # Node: Observed Data
    draw_box(ax, 0.10, 0.07, 0.26, 0.11, r"$d(\hat{n}) = \tilde{T}(\hat{n}) + n(\hat{n}), \quad n \sim \mathcal{N}(0, \mathbf{N})$", "0.3", title=r"Observed Sky $d$")
    draw_arrow(ax, 0.23, 0.25, 0.23, 0.18, color="0.4")

    # Section 2: 4-Block Gibbs Sampler (Right side: 0.49 to 0.98)
    bg_gibbs = patches.FancyBboxPatch(
        (0.48, 0.04), 0.50, 0.92,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=0.8,
        edgecolor="0.8",
        facecolor="0.97",
        zorder=1
    )
    ax.add_patch(bg_gibbs)
    ax.text(0.50, 0.92, "(b) Four-Block Joint Gibbs Sampler", fontsize=8.5, fontweight="bold", color="0.2", zorder=4)

    # Block 1: C_l
    draw_box(ax, 0.51, 0.70, 0.44, 0.16,
             r"$C_\ell \sim \mathrm{InvGamma}\left(\frac{2\ell+1}{2}-1, \frac{S_\ell}{2}\right)$" + "\n" +
             r"Exact conjugate draw on $S_\ell = \sum_m |a_{\ell m}|^2$",
             paper_style.COL_ALM, title=r"Block 1: CMB Spectrum $C_\ell^{TT}$")

    # Block 2: a_lm
    draw_box(ax, 0.51, 0.49, 0.44, 0.16,
             r"$a_{\ell m} \sim \mathcal{P}(a_{\ell m} \mid C_\ell, \phi, d)$" + "\n" +
             r"MCLMC / HMC with reverse-mode AD through SHT",
             paper_style.COL_ALM, title=r"Block 2: Temperature Field $a_{\ell m}$")

    # Block 3: phi
    draw_box(ax, 0.51, 0.28, 0.44, 0.16,
             r"$\phi \sim \mathcal{P}(\phi \mid C_L^{\phi\phi}, a_{\ell m}, d)$" + "\n" +
             r"Curved-sky HMC with gradient $\nabla_\phi \log \mathcal{L}$ through $\mathcal{L}(\phi)$",
             paper_style.COL_PHI, title=r"Block 3: Lensing Potential $\phi_{LM}$")

    # Block 4: C_L^phiphi
    draw_box(ax, 0.51, 0.07, 0.44, 0.16,
             r"$C_L^{\phi\phi} \sim \mathrm{InvGamma}\left(\frac{\nu_0 + 2L+1}{2}-1, \frac{\nu_0 C_{L,0}^{\phi\phi} + S_L^\phi}{2}\right)$" + "\n" +
             r"Exact conjugate draw with proper prior ($\nu_0=6$)",
             paper_style.COL_CLPP, title=r"Block 4: Deflection Spectrum $C_L^{\phi\phi}$")

    # Cycle arrows linking the blocks
    arrow_col = paper_style.COL_AWARE
    # 1 -> 2
    draw_arrow(ax, 0.73, 0.70, 0.73, 0.65, color=arrow_col, style="-|>")
    # 2 -> 3
    draw_arrow(ax, 0.73, 0.49, 0.73, 0.44, color=arrow_col, style="-|>")
    # 3 -> 4
    draw_arrow(ax, 0.73, 0.28, 0.73, 0.23, color=arrow_col, style="-|>")

    # 4 loop back to 1 (curved arrow around right side)
    ax.annotate(
        "",
        xy=(0.95, 0.78), xycoords="data",
        xytext=(0.95, 0.15), textcoords="data",
        arrowprops={
            "arrowstyle": "-|>",
            "color": arrow_col,
            "lw": 1.4,
            "connectionstyle": "arc3,rad=-0.4",
            "mutation_scale": 10,
        },
        zorder=3
    )
    ax.text(0.99, 0.47, "Gibbs\nSweep", fontsize=7, color=arrow_col,
            fontweight="bold", ha="center", va="center", rotation=-90, zorder=4)

    out_path = os.path.join(outdir, "schematic_forward_model.pdf")
    paper_style.save(fig, out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default="/cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure_schematic",
        help="Directory to write schematic vector PDF.",
    )
    args = parser.parse_args()
    make_schematic(args.outdir)


if __name__ == "__main__":
    main()
