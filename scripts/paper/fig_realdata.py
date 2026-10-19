"""Real-data figure (ROADMAP T1.2a): the joint posterior on Planck temperature at lmax = 64.

(a) posterior D_l^TT per multipole (median and 68 % band) on SMICA, and on PR4
    SEVEM if given, with the Planck 2018 TT spectrum and its errors;
(b) the C_L^phiphi posterior per bin over its own prior centre (A_phi x the
    physical fiducial), at A_phi = 3000 and A_phi = 1: 1 means the data left the
    prior where it was.

Reads realdata_summary.py outputs. Caption requirements (plots/STORY.md): the
l < 10 bins are prior-dominated at nu = 30 (f_sky(2l+1) = 4-14 modes per l
against 30 prior dof); the posterior C_l is unlensed, Planck's lensed; at
A_phi = 1 the C_L^phiphi posterior returns its prior (physical S/N 0.002).

Usage:
  PYTHONPATH=diffcmb:scripts:scripts/paper .venv/bin/python scripts/paper/fig_realdata.py \
      --a1 results/analysis/realdata_smica_l64_A1_n30_clnu30/summary.npz \
      --a3000 results/analysis/realdata_smica_l64_A3000_n30_clnu30/summary.npz \
      --sevem results/analysis/realdata_sevemR4_l64_A1_n30_clnu30/summary.npz \
      --outdir <plots>/figure_realdata
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
import paper_style as ps  # noqa: E402

BULK_FROM_ELL = 10  # the prior-dominated bins below this are left out of the stat box


def _bin_centres(columns):
    out = []
    for c in columns:
        if c.startswith("l="):
            out.append(float(c[2:]))
        else:
            lo, hi = c.strip("[)").split(",")
            out.append(0.5 * (float(lo) + float(hi) - 1))
    return np.array(out)


def _bulk(columns):
    """Bins at l >= BULK_FROM_ELL, the single-l band-edge column excluded."""
    return np.array([not c.startswith("l=") and int(c.strip("[").split(",")[0]) >= BULK_FROM_ELL
                     for c in columns])


def panel_tt(ax, s, s_alt):
    ell = s["ell"]
    ax.fill_between(ell, s["C_l^TT/dl_ell_q16"], s["C_l^TT/dl_ell_q84"], color=ps.COL_AWARE,
                    alpha=0.25, lw=0, step="mid")
    ax.plot(ell, s["C_l^TT/dl_ell_q50"], color=ps.COL_AWARE, lw=1.0, drawstyle="steps-mid",
            label="joint posterior, SMICA")
    if s_alt is not None:
        ax.plot(s_alt["ell"], s_alt["C_l^TT/dl_ell_q50"], color=ps.COL_ALTMAP, lw=1.0,
                drawstyle="steps-mid", label="joint posterior, PR4 SEVEM")
    ax.errorbar(ell, s["planck_dl_ell"], yerr=[s["planck_dl_lo"], s["planck_dl_hi"]],
                fmt="o", ms=1.8, lw=0.6, color=ps.COL_TRUTH, label="Planck 2018 TT")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{TT}/2\pi\ [\mu{\rm K}^2]$")
    ax.set_xlim(1.5, ell[-1] + 0.5)
    ax.set_ylim(0, None)
    r = s["C_l^TT/planck_over_median"][_bulk(list(s["columns"]))]
    ps.stat_box(ax, rf"Planck / posterior, $\ell \geq {BULK_FROM_ELL}$: {r.min():.2f} to {r.max():.2f}",
                loc="lower right")
    ax.legend(loc="upper center", frameon=False, fontsize=6.5)


def panel_pp(ax, s_hi, s_lo):
    cols = list(s_hi["columns"])
    x = _bin_centres(cols)
    ax.axhline(1.0, color=ps.COL_NULL, lw=0.8, ls="--", zorder=0)
    for s, marker, face, dx in ((s_hi, "o", ps.COL_CLPP, -0.6), (s_lo, "s", "white", 0.6)):
        q16, q50, q84 = (s[f"C_L^phiphi/over_prior_q{q}"] for q in (16, 50, 84))
        ax.errorbar(x + dx, q50, yerr=[q50 - q16, q84 - q50], fmt=marker, ms=3.5, lw=0.8,
                    color=ps.COL_CLPP, mfc=face, label=rf"$A_\phi = {float(s['phi_amplitude']):g}$")
    ax.set_xlabel(r"$L$ (bin centre)")
    ax.set_ylabel(r"$C_L^{\phi\phi}$ / prior centre")
    r = s_hi["C_L^phiphi/over_prior_q50"]
    ps.stat_box(ax, rf"$A_\phi = {float(s_hi['phi_amplitude']):g}$: {r.min():.2f} to {r.max():.2f}",
                loc="lower left")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a1", required=True, help="summary.npz of the SMICA A_phi = 1 run")
    p.add_argument("--a3000", required=True, help="summary.npz of the SMICA A_phi = 3000 run")
    p.add_argument("--sevem", default=None, help="summary.npz of a PR4 SEVEM run (optional)")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()
    s1, s3 = np.load(args.a1), np.load(args.a3000)
    for s, name in ((s1, "--a1"), (s3, "--a3000")):
        if len(s["collapsed_ell"]):
            raise SystemExit(f"{name}: collapsed C_l at l = {list(s['collapsed_ell'])}; not a posterior")
    s_alt = np.load(args.sevem) if args.sevem else None
    ps.apply()
    os.makedirs(args.outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    panel_tt(ax, s1, s_alt)
    ps.save(fig, os.path.join(args.outdir, "realdata_cltt.pdf"))
    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    panel_pp(ax, s3, s1)
    ps.save(fig, os.path.join(args.outdir, "realdata_clpp.pdf"))


if __name__ == "__main__":
    main()
