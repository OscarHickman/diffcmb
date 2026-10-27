"""LCDM parameters from the joint chains (ROADMAP T1.4): a lensing-marginalised low-l TT likelihood.

Reads scripts/br_lcdm.py outputs (results/analysis/br_lcdm/), never the chains:
  {smica,sevemR4}_A1_pooled8.npz      real data, each noise-marginalised over 8 draws;
  d5_calibration.npz                  flat-prior full-sky ensemble, --target lcdm;
  realsky_strict_calibration.npz      strict real-sky (masked) ensemble, --target hier.

Panels (one PDF each, single column):
  lcdm_posterior            68/95 % contours of (A_30, n_s), A_30 = C_30/C_30^fid, for SMICA and
                            PR4 SEVEM, with Planck 2018's point;
  lcdm_chain_vs_oracle_ns   per sky, the chain-based posterior mean of n_s against the oracle's
  lcdm_chain_vs_oracle_A    (the same likelihood from the true unlensed a_lm), both ensembles:
                            the chains reproduce what knowing the sky would give, without bias.
Refuses a posterior that touches its grid and calibration files without oracle rows.
Caption facts are printed, not drawn.

Usage:
  PYTHONPATH=diffcmb:scripts:scripts/paper .venv/bin/python scripts/paper/fig_lcdm.py \\
      --indir results/analysis/br_lcdm --outdir <plots>/figure9
"""
import argparse
import os
import sys

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps  # noqa: E402

ps.apply()
import matplotlib.pyplot as plt  # noqa: E402

# Planck 2018 TT,TE,EE+lowE+lensing (Planck 2018 VI, Table 2): ln(1e10 A_s) = 3.044, tau = 0.0544,
# n_s = 0.9649. In br_lcdm's units A_30 = (A_s e^-2tau) / (2e-9 e^-0.132) x r_30(n_s) = 1.0743.
PLANCK = (1.0743, 0.9649)
EDGE_WARN = 1e-3
MAPS = (("smica", "SMICA", ps.COL_AWARE), ("sevemR4", "PR4 SEVEM", ps.COL_ALTMAP))
ENSEMBLES = (("d5_calibration", "full sky", "o", None), ("realsky_strict_calibration", "masked", "s", "white"))


def edge_fraction(p):
    return max(p[0].max(), p[-1].max(), p[:, 0].max(), p[:, -1].max()) / p.max()


def hpd_levels(p, masses=(0.68, 0.95)):
    """Density thresholds enclosing the given posterior masses."""
    flat = np.sort(p.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    return [flat[np.searchsorted(cum, m)] for m in masses]


def compare_planck(q):
    """Posterior moments and Planck's chi^2_2 p and enclosing mass."""
    p, a, n = q["post"], q["A_grid"], q["ns_grid"]
    aa, nn = np.meshgrid(a, n, indexing="ij")
    ma, mn = (p * aa).sum(), (p * nn).sum()
    cov = np.array([[(p * (aa - ma) ** 2).sum(), (p * (aa - ma) * (nn - mn)).sum()],
                    [(p * (aa - ma) * (nn - mn)).sum(), (p * (nn - mn) ** 2).sum()]])
    dv = np.array(PLANCK) - (ma, mn)
    i, j = np.argmin(abs(a - PLANCK[0])), np.argmin(abs(n - PLANCK[1]))
    return {"A": ma, "A_sd": np.sqrt(cov[0, 0]), "ns": mn, "ns_sd": np.sqrt(cov[1, 1]),
            "corr": cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]),
            "p": chi2.sf(dv @ np.linalg.solve(cov, dv), 2), "mass": p[p > p[i, j]].sum()}


def load(indir):
    real = {m: np.load(os.path.join(indir, f"{m}_A1_pooled8.npz")) for m, *_ in MAPS}
    cal = {e: np.load(os.path.join(indir, f"{e}.npz")) for e, *_ in ENSEMBLES}
    for m, q in real.items():
        if edge_fraction(q["post"]) > EDGE_WARN:
            raise SystemExit(f"{m}: the posterior touches its grid (boundary density "
                             f"{edge_fraction(q['post']):.2g} of the peak); not a posterior")
    for e, c in cal.items():
        if "oracle" not in c.files:
            raise SystemExit(f"{e}.npz has no oracle rows; rerun scripts/br_lcdm.py --ensemble")
        if max(c["rows"][:, 7].max(), c["oracle"][:, 7].max()) > EDGE_WARN:
            raise SystemExit(f"{e}: posteriors touch the grid; not a calibration")
    return real, cal


def plot_posterior(real, outpath):
    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    for m, label, col in MAPS:
        q = real[m]
        lv = hpd_levels(q["post"])
        if m == "smica":  # the second map as lines only, so both stay readable
            for lo, hi, alpha in ((lv[1], lv[0], 0.15), (lv[0], q["post"].max(), 0.35)):
                ax.contourf(q["A_grid"], q["ns_grid"], q["post"].T, levels=[lo, hi], colors=[col], alpha=alpha)
        ax.contour(q["A_grid"], q["ns_grid"], q["post"].T, levels=[lv[1], lv[0]], colors=[col],
                   linewidths=[0.7, 1.1])
        ax.plot([], [], color=col, lw=1.1, label=label)
    ax.plot(*PLANCK, marker="*", ms=8, color=ps.COL_TRUTH, ls="none", label="Planck 2018")
    ax.set_xlabel(r"$A_{30}\equiv C_{30}/C_{30}^{\rm fid}$")
    ax.set_ylabel(r"$n_s$")
    a0 = real["smica"]
    sel = a0["post"].sum(1) > 1e-4 * a0["post"].sum(1).max()
    seln = a0["post"].sum(0) > 1e-4 * a0["post"].sum(0).max()
    ax.set_xlim(a0["A_grid"][sel][0], a0["A_grid"][sel][-1])
    ax.set_ylim(a0["ns_grid"][seln][0], a0["ns_grid"][seln][-1])
    ax.legend(loc="upper left", frameon=False, fontsize=6.5)
    c = [compare_planck(real[m]) for m, *_ in MAPS]
    ps.stat_box(ax, rf"Planck: $p={c[0]['p']:.2f}$ / ${c[1]['p']:.2f}$", loc="lower left")
    ps.save(fig, outpath)
    return c


def plot_chain_vs_oracle(cal, col, label, outpath):
    """Posterior mean, chains against the oracle, in units of the oracle sd from the truth."""
    truth = 1.0 if col == 0 else 0.965  # every calibration sky's truth: A_30 = 1, n_s = NS_FID
    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    stats = []
    lim = 0
    for e, name, marker, face in ENSEMBLES:
        r, o = cal[e]["rows"], cal[e]["oracle"]
        x = (o[:, col] - truth) / o[:, col + 1]
        y = (r[:, col] - truth) / o[:, col + 1]
        lim = max(lim, np.abs(x).max(), np.abs(y).max())
        ax.plot(x, y, marker, ms=3.2, color=ps.COL_AWARE, mfc=face or ps.COL_AWARE, ls="none", label=name)
        dz = y - x
        stats.append((dz.mean(), np.sqrt(np.mean(dz ** 2)), np.corrcoef(r[:, col], o[:, col])[0, 1]))
    lim = np.ceil(lim + 0.3)
    ax.plot([-lim, lim], [-lim, lim], color=ps.COL_NULL, lw=0.8, zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(rf"oracle: ({label} $-$ truth) / oracle sd")
    ax.set_ylabel(rf"chains: ({label} $-$ truth) / oracle sd")
    ax.legend(loc="upper left", frameon=False, fontsize=6.5)
    ps.stat_box(ax, rf"offset ${stats[0][0]:+.2f}$ / ${stats[1][0]:+.2f}$", loc="lower right")
    ps.save(fig, outpath)
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--indir", required=True)
    p.add_argument("--outdir", required=True)
    a = p.parse_args()
    real, cal = load(a.indir)
    os.makedirs(a.outdir, exist_ok=True)
    out = lambda name: os.path.join(a.outdir, f"{name}.pdf")  # noqa: E731
    c = plot_posterior(real, out("lcdm_posterior"))
    s_ns = plot_chain_vs_oracle(cal, 2, r"$n_s$", out("lcdm_chain_vs_oracle_ns"))
    s_a = plot_chain_vs_oracle(cal, 0, r"$A_{30}$", out("lcdm_chain_vs_oracle_A"))

    print("\nCAPTION FACTS (do not draw these into the panels):")
    print(f"  l = {int(real['smica']['lmin'])}-{int(real['smica']['lmax_use']) - 1}; Planck 2018 at "
          f"A_30 = {PLANCK[0]}, n_s = {PLANCK[1]}; flat priors on (A_30, n_s)")
    for (_, label, _), x in zip(MAPS, c):
        print(f"  {label}: A_30 {x['A']:.3f} +- {x['A_sd']:.3f}, n_s {x['ns']:.3f} +- {x['ns_sd']:.3f}, "
              f"corr {x['corr']:+.2f}; Planck chi2_2 p {x['p']:.2f}, at posterior mass {x['mass']:.2f}")
    for (e, name, *_), sn, sa in zip(ENSEMBLES, s_ns, s_a):
        print(f"  {name} ({e}): chain - oracle n_s mean {sn[0]:+.2f} rms {sn[1]:.2f} corr {sn[2]:.3f}; "
              f"A_30 mean {sa[0]:+.2f} rms {sa[1]:.2f} corr {sa[2]:.3f}")


if __name__ == "__main__":
    main()
