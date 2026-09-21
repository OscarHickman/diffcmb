"""Paper schematic -- Forward model and 4-block differentiable Gibbs sampler.

Produces a publication-grade vector PDF:
  schematic_forward_model.pdf
"""

import argparse
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import paper_style


def draw_box(ax, x, y, w, h, text, color, alpha=0.08, zorder=2):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.1,
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
        zorder=zorder
    )
    ax.add_patch(rect)
    rect_border = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.1,
        edgecolor=color,
        facecolor="none",
        zorder=zorder + 1
    )
    ax.add_patch(rect_border)

    cx = x + w / 2
    cy = y + h / 2
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=6.3, color="0.1", zorder=zorder + 2, linespacing=1.28)


def draw_arrow(ax, x1, y1, x2, y2, color="0.4", width=1.1, zorder=3, style="-|>"):
    ax.annotate(
        "",
        xy=(x2, y2), xycoords="data",
        xytext=(x1, y1), textcoords="data",
        arrowprops={
            "arrowstyle": style,
            "color": color,
            "lw": width,
            "shrinkA": 2, "shrinkB": 2,
            "mutation_scale": 9,
        },
        zorder=zorder
    )


def make_schematic(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    paper_style.apply()

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Section 1: Forward Model (Left side: 0.02 to 0.46)
    bg_fwd = patches.FancyBboxPatch(
        (0.02, 0.03), 0.44, 0.94,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=0.8,
        edgecolor="0.82",
        facecolor="0.98",
        zorder=1
    )
    ax.add_patch(bg_fwd)
    ax.text(0.04, 0.935, "(a) Generative Forward Model", fontsize=8.2, fontweight="bold", color="0.2", zorder=4)

    # Row 1: Priors
    draw_box(ax, 0.045, 0.74, 0.18, 0.14,
             r"$\mathbf{C_\ell^{TT}\ \mathbf{Prior}}$" + "\n" + r"$\mathcal{P}(C_\ell^{TT})$",
             paper_style.COL_ALM)
    draw_box(ax, 0.255, 0.74, 0.18, 0.14,
             r"$\mathbf{C_L^{\phi\phi}\ \mathbf{Prior}}$" + "\n" + r"$\mathcal{P}(C_L^{\phi\phi}) \propto \mathrm{InvGamma}(\nu_0)$",
             paper_style.COL_CLPP)

    draw_arrow(ax, 0.135, 0.74, 0.135, 0.67, color=paper_style.COL_ALM)
    draw_arrow(ax, 0.345, 0.74, 0.345, 0.67, color=paper_style.COL_CLPP)

    # Row 2: Latent fields
    draw_box(ax, 0.045, 0.48, 0.18, 0.19,
             r"$\mathbf{Primary\ } T(\hat{n})$" + "\n" +
             r"$a_{\ell m} \sim \mathcal{N}(0, C_\ell)$" + "\n" +
             r"$T(\hat{n}) = \mathrm{SHT}(a_{\ell m})$",
             paper_style.COL_ALM)
    draw_box(ax, 0.255, 0.48, 0.18, 0.19,
             r"$\mathbf{Lensing\ } \phi(\hat{n})$" + "\n" +
             r"$\phi_{LM} \sim \mathcal{N}(0, C_L^{\phi\phi})$" + "\n" +
             r"$\mathbf{d}(\hat{n}) = \nabla \phi(\hat{n})$",
             paper_style.COL_PHI)

    draw_arrow(ax, 0.135, 0.48, 0.19, 0.41, color="0.4")
    draw_arrow(ax, 0.345, 0.48, 0.29, 0.41, color="0.4")

    # Row 3: Lensing Deflection
    draw_box(ax, 0.08, 0.25, 0.32, 0.16,
             r"$\mathbf{Lensing\ Operator\ } \mathcal{L}(\phi)$" + "\n" +
             r"$\tilde{T}(\hat{n}) = T(\hat{n} + \nabla \phi(\hat{n}))$" + "\n" +
             r"(curved-sky HEALPix deflection)",
             paper_style.COL_AWARE)

    draw_arrow(ax, 0.24, 0.25, 0.24, 0.20, color="0.4")

    # Row 4: Observed Data
    draw_box(ax, 0.08, 0.07, 0.32, 0.13,
             r"$\mathbf{Observed\ Sky\ } d$" + "\n" +
             r"$d(\hat{n}) = \tilde{T}(\hat{n}) + n(\hat{n}), \quad n \sim \mathcal{N}(0, \mathbf{N})$",
             "0.3")

    # Section 2: 4-Block Gibbs Sampler (Right side: 0.48 to 0.98)
    bg_gibbs = patches.FancyBboxPatch(
        (0.48, 0.03), 0.50, 0.94,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=0.8,
        edgecolor="0.82",
        facecolor="0.98",
        zorder=1
    )
    ax.add_patch(bg_gibbs)
    ax.text(0.50, 0.935, "(b) Four-Block Joint Gibbs Sampler", fontsize=8.2, fontweight="bold", color="0.2", zorder=4)

    # 4 Blocks vertically stacked with comfortable margins
    bw = 0.41
    bx = 0.495
    bh = 0.18

    # Block 1: C_l
    draw_box(ax, bx, 0.725, bw, bh,
             r"$\mathbf{Block\ 1:\ CMB\ Spectrum\ } C_\ell^{TT}$" + "\n" +
             r"$C_\ell \sim \mathrm{InvGamma}\left(\frac{2\ell+1}{2}-1, \frac{S_\ell}{2}\right)$" + "\n" +
             r"Exact conjugate draw on $S_\ell = \sum_m |a_{\ell m}|^2$",
             paper_style.COL_ALM)

    # Block 2: a_lm
    draw_box(ax, bx, 0.505, bw, bh,
             r"$\mathbf{Block\ 2:\ Temperature\ Field\ } a_{\ell m}$" + "\n" +
             r"$a_{\ell m} \sim \mathcal{P}(a_{\ell m} \mid C_\ell, \phi, d)$" + "\n" +
             r"MCLMC / HMC with reverse-mode AD through SHT",
             paper_style.COL_ALM)

    # Block 3: phi
    draw_box(ax, bx, 0.285, bw, bh,
             r"$\mathbf{Block\ 3:\ Lensing\ Potential\ } \phi_{LM}$" + "\n" +
             r"$\phi \sim \mathcal{P}(\phi \mid C_L^{\phi\phi}, a_{\ell m}, d)$" + "\n" +
             r"Curved-sky HMC with gradient $\nabla_\phi \log \mathcal{L}$ through $\mathcal{L}(\phi)$",
             paper_style.COL_PHI)

    # Block 4: C_L^phiphi
    draw_box(ax, bx, 0.065, bw, bh,
             r"$\mathbf{Block\ 4:\ Deflection\ Spectrum\ } C_L^{\phi\phi}$" + "\n" +
             r"$C_L^{\phi\phi} \sim \mathrm{InvGamma}\left(\frac{\nu_0 + 2L+1}{2}-1, \frac{\nu_0 C_{L,0}^{\phi\phi} + S_L^\phi}{2}\right)$" + "\n" +
             r"Exact conjugate draw with proper prior ($\nu_0 = 6$)",
             paper_style.COL_CLPP)

    # Connecting vertical arrows
    cx_gibbs = bx + bw / 2
    draw_arrow(ax, cx_gibbs, 0.725, cx_gibbs, 0.685, color=paper_style.COL_AWARE)
    draw_arrow(ax, cx_gibbs, 0.505, cx_gibbs, 0.465, color=paper_style.COL_AWARE)
    draw_arrow(ax, cx_gibbs, 0.285, cx_gibbs, 0.245, color=paper_style.COL_AWARE)

    # Curved loop from Block 4 back to Block 1 on the right margin (rad > 0 curves outwards to the right!)
    ax.annotate(
        "",
        xy=(bx + bw + 0.012, 0.815), xycoords="data",
        xytext=(bx + bw + 0.012, 0.155), textcoords="data",
        arrowprops={
            "arrowstyle": "-|>",
            "color": paper_style.COL_AWARE,
            "lw": 1.3,
            "connectionstyle": "arc3,rad=0.32",
            "mutation_scale": 10,
        },
        zorder=3
    )
    ax.text(0.965, 0.485, "Gibbs Sweep", fontsize=6.8, color=paper_style.COL_AWARE,
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
