"""Paper figure 2 -- a lensing-blind fit is biased by exactly the lensing; the joint fit is not.

Rebuilt 2026-09-23 (ROADMAP D3). The earlier version compared single chains at
lmax=128/192 made with the bilinear lensing operator, whose interpolation
smoothing -- not lensing -- produced almost all of the "bias" it showed. This
version uses the lmax=64 ensemble, where exactness is certified, with every
sky lensed by the exact operator, and pairs each lensing-aware chain
(chain_rNNN.npz) with a lensing-blind fit to the IDENTICAL data
(blind_rNNN.npz, coverage_ensemble_chain.py --blind).

Two panels:

  (a) bias_by_bin.pdf   per l-bin, mean +/- SEM over skies of the fractional
                        bias of the posterior-mean C_l^TT, blind vs aware,
                        with the lensing each sky actually received (the
                        EXPECTATION for the blind fit) drawn as a reference.
  (b) bias_per_sky.pdf  per sky, in the bin with the largest lensing effect:
                        bias against that sky's own lensing effect. The blind
                        fit tracks the diagonal (it absorbs the lensing); the
                        aware fit sits on zero.

REFERENCE. The bias of a chain is measured against the posterior mean a
CORRECT unlensed-sky model would have: under the flat C_l prior the Block 1
conditional InvGamma(k/2-1, S/2) has mean S/(k-4), so the reference is
S_l(alm_true)/(k_l - 4) per multipole, summed over a bin. (Using the realized
S/k or the fiducial adds an l-dependent offset common to both chains that
buries the signal; compare_cl_bias_reduction.py records the history.)

EXPECTED LENSING. For each sky the in-band power of its lensed truth map
(exact operator, same phi, noise-free) over that of the unlensed truth, minus
one. That is what an unlensed model fitted to lensed data should report, so the
blind curve should land on it; a blind curve that did not would indict the
comparison, not the sampler.

Usage:
  PYTHONPATH=diffcmb:scripts:scripts/paper .venv/bin/python scripts/paper/fig2_bias_reduction.py \\
      --indir results/analysis/<ensemble dir with chain_ and blind_ files>
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import healpy as hp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paper_style as ps  # noqa: E402

ps.apply()
import matplotlib.pyplot as plt  # noqa: E402
from aggregate_coverage_ranks import _sl_from_packed  # noqa: E402

from diffcmb.alm_utils import packed_dof_per_multipole, packed_length  # noqa: E402
from diffcmb.lensing import _alm_packed_to_hp, exact_lens_np  # noqa: E402

BINS = [(2, 10), (10, 20), (20, 30), (30, 45), (45, 64)]
DEFAULT_OUT = "/cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure2"


def pairs(indir):
    out = []
    for f in sorted(glob.glob(os.path.join(indir, "chain_r[0-9][0-9][0-9].npz"))):
        b = f.replace("chain_r", "blind_r")
        if os.path.exists(b):
            out.append((f, b))
    if not out:
        raise SystemExit(f"no chain_rNNN / blind_rNNN pairs in {indir}")
    return out


def _check_pair(a, b, fa):
    lmax = int(a["lmax"])
    if a["alm_true_packed"].size != packed_length(lmax):
        raise SystemExit(f"{fa}: pre-2026-09-06 packing, refusing")
    for key in ("alm_true_packed", "phi_true_packed", "cl_true"):
        if not np.array_equal(a[key], b[key]):
            raise SystemExit(f"{fa}: aware and blind chains differ in {key} -- "
                             "not the same sky, the comparison is meaningless")
    op = str(a["lensing_operator"]) if "lensing_operator" in a.files else "bilinear"
    if op != "exact":
        raise SystemExit(f"{fa}: made with the {op} operator; figure 2 needs "
                         "exact-operator chains (ROADMAP D3)")
    return lmax


def sky_biases(fa, fb):
    """Per-bin fractional bias (aware, blind) and the sky's lensing effect."""
    a, b = np.load(fa), np.load(fb)
    lmax = _check_pair(a, b, fa)
    nside = int(a["nside"])
    n_cl = lmax - 2
    k = packed_dof_per_multipole(lmax).astype(float)
    S = _sl_from_packed(np.asarray(a["alm_true_packed"], float), lmax)
    ref = np.zeros(lmax)
    ref[2:] = S[2:] / (k[2:] - 4.0)
    cl_aware = np.zeros(lmax)
    cl_blind = np.zeros(lmax)
    cl_aware[2:] = np.exp(a["alm_samples"][:, :n_cl]).mean(axis=0)
    cl_blind[2:] = np.exp(b["alm_samples"][:, :n_cl]).mean(axis=0)

    alm_hp = _alm_packed_to_hp(np.asarray(a["alm_true_packed"], float), lmax)
    pix = np.arange(hp.nside2npix(nside))
    lensed = exact_lens_np(alm_hp, np.asarray(a["phi_true_packed"], float), nside, lmax, pix)
    unl = hp.alm2map(alm_hp, nside, lmax=lmax - 1)
    p_len = hp.anafast(lensed, lmax=lmax - 1, iter=3)
    p_unl = hp.anafast(unl, lmax=lmax - 1, iter=3)

    rows = []
    for lo, hi in BINS:
        ell = np.arange(lo, min(hi, lmax))
        if ell.size == 0:
            rows.append((np.nan, np.nan, np.nan))
            continue
        w = 2 * ell + 1
        rows.append((np.sum(cl_aware[ell] * w) / np.sum(ref[ell] * w) - 1.0,
                     np.sum(cl_blind[ell] * w) / np.sum(ref[ell] * w) - 1.0,
                     np.sum(p_len[ell] * w) / np.sum(p_unl[ell] * w) - 1.0))
    return np.array(rows) * 100.0, lmax


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", required=True)
    p.add_argument("--outdir", default=DEFAULT_OUT)
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    sky = []
    for fa, fb in pairs(a.indir):
        rows, lmax = sky_biases(fa, fb)
        sky.append(rows)
    sky = np.array(sky)                       # (n_sky, n_bin, [aware, blind, expected])
    n = sky.shape[0]
    keep = [i for i, (lo, _) in enumerate(BINS) if lo < lmax]
    centres = np.array([0.5 * (lo + min(hi, lmax)) for lo, hi in BINS])[keep]
    mean = np.nanmean(sky, axis=0)[keep]
    sem = (np.nanstd(sky, axis=0, ddof=1) / np.sqrt(n))[keep]

    # (a) bias by bin ------------------------------------------------------
    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    ax.axhline(0.0, color="0.3", lw=0.6)
    ax.plot(centres, mean[:, 2], color=ps.COL_TRUTH, lw=0.9, ls="--",
            label="lensing received (expected blind)")
    ax.errorbar(centres - 0.6, mean[:, 1], sem[:, 1], fmt="o", ms=3.5, capsize=2,
                color=ps.COL_BLIND, label="lensing-blind")
    ax.errorbar(centres + 0.6, mean[:, 0], sem[:, 0], fmt="s", ms=3.5, capsize=2,
                color=ps.COL_AWARE, label="lensing-aware (this work)")
    ax.set_xlabel(r"multipole $\ell$ (bin centre)")
    ax.set_ylabel(r"bias in $C_\ell^{TT}$ (%)")
    ax.set_xlim(0, lmax)
    ax.legend(loc="best")
    blind_abs = np.nanmean(np.abs(mean[:, 1]))
    aware_abs = np.nanmean(np.abs(mean[:, 0]))
    ps.stat_box(ax, f"{n} skies\nreduction {100 * (1 - aware_abs / blind_abs):.0f}%",
                loc="lower left")
    ps.save(fig, os.path.join(a.outdir, "bias_by_bin.pdf"))

    # (b) per sky, bin with the largest lensing effect ----------------------
    j = keep[int(np.nanargmax(np.abs(mean[:, 2])))]
    x = sky[:, j, 2]
    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    lim = 1.15 * np.nanmax(np.abs(np.concatenate([x, sky[:, j, 1]])))
    ax.plot([-lim, lim], [-lim, lim], color=ps.COL_NULL, lw=0.8)
    ax.axhline(0.0, color="0.3", lw=0.6)
    ax.plot(x, sky[:, j, 1], "o", ms=3.5, color=ps.COL_BLIND, label="lensing-blind")
    ax.plot(x, sky[:, j, 0], "s", ms=3.5, color=ps.COL_AWARE, label="lensing-aware")
    jlo, jhi = BINS[j][0], min(BINS[j][1], lmax)
    ax.set_xlabel(rf"lensing received, $\ell\in[{jlo},{jhi})$ (%)")
    ax.set_ylabel(r"bias in $C_\ell^{TT}$ (%)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.legend(loc="upper left")
    r_blind = float(np.corrcoef(x, sky[:, j, 1])[0, 1])
    ps.stat_box(ax, rf"$r_{{\rm blind}}={r_blind:.2f}$", loc="lower right")
    ps.save(fig, os.path.join(a.outdir, "bias_per_sky.pdf"))

    print("\nCAPTION FACTS (do not draw these into the panels):")
    print(f"  {n} sky pairs from {a.indir}; exact operator; lmax={lmax}")
    for i in keep:
        lo, hi = BINS[i]
        print(f"  [{lo:2d},{min(hi, lmax):2d})  aware {np.nanmean(sky[:, i, 0]):+6.3f}%  "
              f"blind {np.nanmean(sky[:, i, 1]):+6.3f}%  expected {np.nanmean(sky[:, i, 2]):+6.3f}%"
              f"  (SEM aware {np.nanstd(sky[:, i, 0], ddof=1) / np.sqrt(n):.3f})")
    print(f"  mean |bias| over bins: blind {blind_abs:.3f}%, aware {aware_abs:.3f}% "
          f"-> reduction {100 * (1 - aware_abs / blind_abs):.1f}%")
    print(f"  panel (b) bin [{jlo},{jhi}): corr(blind bias, lensing received) = {r_blind:.3f}")


if __name__ == "__main__":
    main()
