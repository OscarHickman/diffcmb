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

Sizes and fonts come from a JOURNAL preset (default PRD, override with the
environment variable DIFFCMB_JOURNAL=MNRAS|JCAP). FIG_1COL fits one column,
FIG_2COL spans a figure* ; FIG_MAP is one of three Mollweide panels across a
figure*. Panels are saved at their final printed size with no rescaling in
LaTeX, so font sizes here are the font sizes on the page.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (column width, figure* width) in inches, and the body font to match.
# PRD/revtex4-2: 8.6 cm column, 17.8 cm text, Computer Modern.
# MNRAS: 84 mm column, 174 mm text, Times.  JCAP: single column ~15.9 cm, CM.
_JOURNALS = {
    "PRD": {"w1": 3.375, "w2": 7.0, "serif": ["cmr10"], "math": "cm"},
    "MNRAS": {"w1": 3.30, "w2": 6.85, "serif": ["Nimbus Roman", "STIXGeneral"], "math": "stix"},
    "JCAP": {"w1": 3.10, "w2": 6.25, "serif": ["cmr10"], "math": "cm"},
}
JOURNAL = os.environ.get("DIFFCMB_JOURNAL", "PRD").upper()
_J = _JOURNALS[JOURNAL]

# Printed panel sizes in inches. Do not rescale with \includegraphics[width=...]
# beyond the column width, or the type sizes below stop being what lands.
FIG_1COL = (_J["w1"], 2.45)
FIG_1COL_TALL = (_J["w1"], 2.95)
FIG_2COL = (_J["w2"], 2.60)
FIG_MAP = (_J["w2"] / 3.0 - 0.04, 1.62)

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
        "font.serif": _J["serif"],
        "mathtext.fontset": _J["math"],
        "axes.unicode_minus": False,      # cmr10 has no U+2212 glyph
        "axes.formatter.use_mathtext": True,
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        # Constrained layout + standard bbox: the saved PDF is exactly the
        # figsize above, so the page size of a panel is known, not cropped to
        # whatever "tight" decides. (tight_layout() must not be called on top.)
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "savefig.bbox": "standard",
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
