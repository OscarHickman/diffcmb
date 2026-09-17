"""Shared figure style for the DiffCMB paper panels.

One place for every typographic and colour decision, so the four figures read
as one paper rather than four scripts. Import `apply()` at the top of a figure
script and use `COL_*` / `FIG_*` instead of hardcoding.

TWO RULES THAT ARE NOT COSMETIC, both from plots/STORY.md's panel convention:

1. **No prose inside a panel.** No `suptitle`, no `fig.text` footnote, no
   explanatory sentence baked into the image. Everything a reader needs beyond
   the axes belongs in the LaTeX caption, where it can be copyedited, sized
   with the document and read by a screen reader. Burnt-in text also rots: the
   figure1 footer asserted "Block 4 OFF" while the script loaded a Block-4-ON
   ensemble, and no test could have caught it because it was a picture of a
   sentence. Panel titles are allowed only where a panel would otherwise be
   ambiguous in a multi-panel figure, and are kept to a few words.

2. **Numbers that appear in the text appear in the panel too**, via
   `stat_box`, not as a title. A referee checking a claim should not have to
   cross-reference the caption to find the value the panel demonstrates.

Sizes assume a two-column journal (MNRAS/AAS): FIG_1COL fits one column,
FIG_2COL spans both. Panels are saved at their final printed size with no
rescaling in LaTeX, so font sizes here are the font sizes on the page.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Printed panel sizes in inches. Do not rescale with \includegraphics[width=...]
# beyond the column width, or the type sizes below stop being what lands.
FIG_1COL = (3.35, 2.60)
FIG_1COL_TALL = (3.35, 3.10)
FIG_2COL = (7.00, 2.90)

# Semantic palette, fixed across all figures so a colour means the same thing
# everywhere. Okabe-Ito derived: distinguishable in the common forms of colour
# blindness and in greyscale print.
COL_AWARE = "#0072B2"     # lensing-aware joint sampler (this work)
COL_BLIND = "#D55E00"     # lensing-blind / Commander-style baseline
COL_PHI = "#0072B2"       # phi field
COL_ALM = "#D55E00"       # alm field
COL_CLPP = "#009E73"      # C_L^phiphi field (the fourth block)
COL_QE = "#555555"        # quadratic-estimator reference curve
COL_NULL = "#BBBBBB"      # null bands, uniform expectation, reference lines
COL_TRUTH = "#000000"

GREY_TEXT = "0.30"


def apply() -> None:
    """Install the paper rcParams. Call once, before creating any figure."""
    plt.rcParams.update({
        # Serif to match a LaTeX body; mathtext in the same family so an
        # axis label and an inline equation in the caption look related.
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.3,
        "patch.linewidth": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,  # embed as TrueType: selectable, searchable text
        "ps.fonttype": 42,
    })


def stat_box(ax, text: str, loc: str = "upper left", xy=None, **kw) -> None:
    """The panel's headline number, in the corner rather than in a title.

    Kept visually quiet (no frame, grey) so it reads as an annotation on the
    data, not as a second title competing with the caption.
    """
    if xy is None:
        xy = {"upper left": (0.03, 0.97), "upper right": (0.97, 0.97),
              "lower left": (0.03, 0.03), "lower right": (0.97, 0.03)}[loc]
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=kw.pop("fontsize", 7), color=kw.pop("color", GREY_TEXT),
            linespacing=1.35, **kw)


def save(fig, path: str) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
