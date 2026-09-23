"""Paper figure -- MCMC health of the production ensemble, all four Gibbs blocks.

Standalone panels (plots/STORY.md panel convention):

  convergence_trace_logp.pdf     log-posterior traces of every chain, each
                                 centred on its own mean (the chains are
                                 different skies, so absolute levels differ)
  convergence_rhat.pdf           rank-normalised, folded split-R-hat per chain
                                 (Vehtari et al. 2021, arXiv:1903.08008) for the
                                 two spectrum blocks, median and 16-84% band
                                 over chains, against the recommended 1.01
  convergence_autocorr_ess.pdf   bulk ESS per multipole for all four blocks:
                                 C_l (Block 1), a_lm (2), phi_LM (3),
                                 C_L^phiphi (4), against the number of draws
                                 per chain that the rank test uses

WHY PER CHAIN. Each chain in the coverage ensemble is a different simulated
sky, so a cross-chain R-hat compares different posteriors and is meaningless.
Every diagnostic here is computed within a chain and then summarised across
chains.

ESS IS AN ESTIMATE. Geyer's initial-monotone-sequence estimator on
rank-normalised draws. Short chains make any ESS estimator noisy and these
estimators need not agree with each other (arXiv:2408.13411), so the paper
quotes them as indicative, not as exact. The line at draws_per_chain shows
where the rank test's draws stop being approximately independent.

Burn-in: the production chains discard `n_burnin` sweeps before saving, so
every saved sample is post-burn-in; nothing further is dropped here.

Usage:
  PYTHONPATH=diffcmb:scripts/paper .venv/bin/python scripts/paper/fig_convergence.py
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import special, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paper_style as ps  # noqa: E402

from diffcmb.alm_utils import packed_length  # noqa: E402
from diffcmb.lensing import _packed_coord_multipole  # noqa: E402

DEFAULT_INDIR = "results/analysis/ens_exact_l64_A3000_n30"
DEFAULT_OUT = "/cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure_convergence"
RHAT_TARGET = 1.01          # Vehtari et al. 2021 recommendation
COL_BLOCK = {1: ps.COL_ALM, 2: "#E69F00", 3: ps.COL_PHI, 4: ps.COL_CLPP}


# ---------------------------------------------------------------------------
# Diagnostics (vectorised over the last axis = coordinates)
# ---------------------------------------------------------------------------

def _rank_normalise(x: np.ndarray) -> np.ndarray:
    """x: (n_draws, n_coord) -> normal scores of the pooled ranks."""
    r = stats.rankdata(x, axis=0)
    return special.ndtri((r - 0.375) / (x.shape[0] + 0.25))


def _split_rhat_plain(z: np.ndarray) -> np.ndarray:
    n = z.shape[0] // 2
    halves = np.stack([z[:n], z[n:2 * n]])            # (2, n, k)
    w = halves.var(axis=1, ddof=1).mean(axis=0)
    b = n * halves.mean(axis=1).var(axis=0, ddof=1)
    var_plus = (n - 1) / n * w + b / n
    return np.sqrt(var_plus / np.where(w > 0, w, np.inf))


def rank_rhat(x: np.ndarray) -> np.ndarray:
    """Rank-normalised split-R-hat, max of bulk and folded (tail) versions."""
    bulk = _split_rhat_plain(_rank_normalise(x))
    folded = _split_rhat_plain(_rank_normalise(np.abs(x - np.median(x, axis=0))))
    return np.maximum(bulk, folded)


def bulk_ess(x: np.ndarray) -> np.ndarray:
    """Geyer initial-monotone-sequence ESS of rank-normalised draws."""
    z = _rank_normalise(x)
    n = z.shape[0]
    z = z - z.mean(axis=0)
    f = np.fft.rfft(z, n=2 * n, axis=0)
    acov = np.fft.irfft(f * np.conj(f), axis=0)[:n]
    rho = acov / np.where(acov[0] > 0, acov[0], np.inf)
    n_pair = (n - 1) // 2
    pairs = rho[0:2 * n_pair:2] + rho[1:2 * n_pair:2]  # (n_pair, k)
    positive = pairs > 0
    first_neg = np.where(positive.all(axis=0), n_pair, np.argmin(positive, axis=0))
    pairs = np.minimum.accumulate(np.where(positive, pairs, 0.0), axis=0)
    keep = np.arange(n_pair)[:, None] < first_neg[None, :]
    tau = -1.0 + 2.0 * np.sum(np.where(keep, pairs, 0.0), axis=0)
    return n / np.maximum(tau, 1.0 / np.log10(max(n, 10)))


# ---------------------------------------------------------------------------

def load(indir: str):
    files = sorted(glob.glob(os.path.join(indir, "chain_r[0-9][0-9][0-9].npz")))
    if not files:
        raise SystemExit(f"no chain_rNNN.npz in {indir}")
    chains = [np.load(f) for f in files]
    lmax = int(chains[0]["lmax"])
    for c in chains:
        if c["phi_samples"].shape[1] != packed_length(lmax):
            raise SystemExit("packed width mismatch: pre-2026-09-06 chain, refusing.")
        if not bool(c["sample_cl_phiphi"]):
            raise SystemExit("Block 4 OFF chain in a Block-4-ON figure, refusing.")
    return chains, lmax


def per_multipole_median(values: np.ndarray, ell_of_coord: np.ndarray, ells) -> np.ndarray:
    return np.array([np.median(values[ell_of_coord == ell]) for ell in ells])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default=DEFAULT_INDIR)
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--rank_thin", type=int, default=10,
                   help="Thinning figure 1's rank test uses (sets the draws/chain line).")
    a = p.parse_args()
    ps.apply()
    os.makedirs(a.outdir, exist_ok=True)

    chains, lmax = load(a.indir)
    n_cl = lmax - 2
    ells = np.arange(2, lmax)
    ell_coord = _packed_coord_multipole(lmax)
    n_ch = len(chains)
    n_sw = chains[0]["logp"].size
    n_burn = int(chains[0]["n_burnin"])
    draws = n_sw // a.rank_thin

    rhat = {1: [], 4: []}
    ess = {1: [], 2: [], 3: [], 4: []}
    for c in chains:
        lncl = c["alm_samples"][:, :n_cl]
        alm = c["alm_samples"][:, n_cl:]
        cpp = c["cl_phiphi_samples"]
        rhat[1].append(rank_rhat(lncl))
        rhat[4].append(rank_rhat(cpp))
        ess[1].append(bulk_ess(lncl))
        ess[4].append(bulk_ess(cpp))
        ess[2].append(per_multipole_median(bulk_ess(alm), ell_coord, ells))
        ess[3].append(per_multipole_median(bulk_ess(c["phi_samples"]), ell_coord, ells))
    rhat = {k: np.array(v) for k, v in rhat.items()}
    ess = {k: np.array(v) for k, v in ess.items()}

    # (a) centred log-posterior traces -----------------------------------
    # All 24 overplotted is an unreadable block; show the ensemble's per-sweep
    # 2.5-97.5% envelope in grey and three individual chains on top of it.
    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    sw = np.arange(n_burn, n_burn + n_sw)
    centred = np.array([c["logp"] - c["logp"].mean() for c in chains])
    lo, hi = np.percentile(centred, [2.5, 97.5], axis=0)
    ax.fill_between(sw, lo, hi, color=ps.COL_NULL, alpha=0.6, lw=0,
                    label=f"{n_ch} chains, 95% range")
    for i, col in zip((0, n_ch // 2, n_ch - 1), ("#0072B2", "#56B4E9", "#003C5E")):
        ax.plot(sw, centred[i], lw=0.45, color=col)
    ax.axhline(0.0, color="0.3", lw=0.6)
    ax.set_xlim(sw[0], sw[-1])
    lim = 1.15 * np.abs(centred).max()
    ax.set_ylim(-lim, 1.35 * lim)
    ax.set_xlabel("Gibbs sweep")
    ax.set_ylabel(r"$\log\mathcal{P}-\langle\log\mathcal{P}\rangle_{\rm chain}$")
    ax.legend(loc="upper left")
    ps.save(fig, os.path.join(a.outdir, "convergence_trace_logp.pdf"))

    # (b) rank-normalised split-R-hat -------------------------------------
    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    for blk, lab in ((1, r"$C_\ell^{TT}$ (Block 1)"), (4, r"$C_L^{\phi\phi}$ (Block 4)")):
        lo, med, hi = np.percentile(rhat[blk], [16, 50, 84], axis=0)
        ax.fill_between(ells, lo, hi, color=COL_BLOCK[blk], alpha=0.22, lw=0)
        ax.plot(ells, med, color=COL_BLOCK[blk], lw=1.1, label=lab)
    ax.axhline(RHAT_TARGET, color="0.35", ls="--", lw=0.8)
    ax.axhline(1.0, color=ps.COL_NULL, lw=0.7)
    ax.set_xlabel(r"multipole $\ell$ or $L$")
    ax.set_ylabel(r"rank-normalised split-$\hat R$")
    ax.set_xlim(2, lmax - 1)
    ax.set_ylim(0.99, None)
    ax.legend(loc="upper left")
    frac = {k: float(np.mean(rhat[k] > RHAT_TARGET)) for k in rhat}
    ps.stat_box(ax, rf"$\hat R>{RHAT_TARGET}$: " + f"{100 * frac[1]:.0f}% / {100 * frac[4]:.0f}%",
                xy=(0.03, 0.70), loc="upper left")
    ps.save(fig, os.path.join(a.outdir, "convergence_rhat.pdf"))

    # (c) bulk ESS per multipole, all four blocks --------------------------
    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    labels = {1: r"$C_\ell^{TT}$", 2: r"$a_{\ell m}$", 3: r"$\phi_{LM}$",
              4: r"$C_L^{\phi\phi}$"}
    for blk in (1, 2, 3, 4):
        lo, med, hi = np.percentile(ess[blk], [16, 50, 84], axis=0)
        ax.fill_between(ells, lo, hi, color=COL_BLOCK[blk], alpha=0.18, lw=0)
        ax.plot(ells, med, color=COL_BLOCK[blk], lw=1.1, label=labels[blk])
    ax.axhline(draws, color="0.35", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlim(2, lmax - 1)
    ax.set_ylim(4, 4000)
    ax.set_xlabel(r"multipole $\ell$ or $L$")
    ax.set_ylabel(f"bulk ESS per chain (of {n_sw})")
    ax.legend(loc="upper left", ncol=4, columnspacing=0.8, handlelength=1.2)
    ax.text(0.5 * lmax, draws * 0.93, f"{draws} rank draws", fontsize=6.5,
            color="0.35", ha="center", va="top")
    ps.save(fig, os.path.join(a.outdir, "convergence_autocorr_ess.pdf"))

    print("\nCAPTION FACTS (do not draw these into the panels):")
    print(f"  ensemble: {a.indir}  ({n_ch} chains, Block 4 ON, nu="
          f"{float(chains[0]['cl_phiphi_prior_nu'])}); {n_sw} saved sweeps/chain "
          f"after {n_burn} burn-in")
    print(f"  R-hat > {RHAT_TARGET}: Block 1 {100 * frac[1]:.1f}%, Block 4 "
          f"{100 * frac[4]:.1f}% of (chain, multipole) pairs; max "
          f"{rhat[1].max():.3f} / {rhat[4].max():.3f}")
    for blk in (1, 2, 3, 4):
        e = ess[blk]
        below = float(np.mean(e < draws))
        print(f"  Block {blk}: median ESS {np.median(e):.0f}, min {e.min():.0f}; "
              f"{100 * below:.0f}% of (chain, multipole) below the {draws} rank draws")


if __name__ == "__main__":
    main()
