"""Paper figure 1 -- the credibility gate: SBC rank histograms + validation power.

Every claim after this figure (Figure 2 bias reduction, Figure 4 joint posterior)
is only as credible as the sampler being validated here. Two standalone panel
PDFs, following the paper13 convention: one file per subplot, combined in LaTeX
with a `figure*` and lettered in the caption prose.

  (a) rank_histograms.pdf   -- pooled phi/alm field ranks, N=24 chains, the
                               calibrated thin=10 (~60 draws/chain) config
                               that is dashboard.md's "no known open defect"
                               status (2026-09-13, jobs 11986716/11986722).
  (b) validation_power.pdf  -- P(detect) vs posterior-mean-shift size at
                               N=24, n_draws=60 -- turns "cal_p passes" into
                               a stated bound rather than implied exactness.

Panel (a) reuses aggregate_coverage_ranks.py's rank-computation helpers
directly (not a reimplementation) so the histogram is built from the exact
ranks the dashboard's cal_p numbers are computed from. Panel (b) reuses
validate_sbc_statistic_power.py's sbc_ranks Monte Carlo model the same way.

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/paper/fig1_validation.py \
      --indir results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_doffix \
      --thin 10 \
      --outdir /cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure1
"""

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aggregate_coverage_ranks import (  # noqa: E402
    binned_power,
    discrete_uniform_p,
    rank_of,
    rank_spread,
)
from validate_sbc_statistic_power import sbc_ranks  # noqa: E402

from diffcmb.alm_utils import packed_length, packed_sizes  # noqa: E402
from diffcmb.samplers import _alm_index_lm  # noqa: E402

ELL_BINS = [(2, 10), (10, 30), (30, 60), (60, 64)]


def collect_ranks(indir, thin):
    """Per-bin ranks for phi_power and alm_power, mirroring aggregate_coverage_ranks.main."""
    files = sorted(f for f in glob.glob(os.path.join(indir, "chain_r*.npz"))
                   if not f.endswith("_ckpt.npz"))
    if not files:
        raise SystemExit(f"no chains in {indir}")

    records = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        lmax = int(d["lmax"])
        n_lncl = lmax - 2
        n_real = lmax * (lmax + 1) // 2 - 3
        n_imag = packed_sizes(lmax)[1]
        L_arr, _m = _alm_index_lm(lmax, n_real, n_imag)

        alm_s = d["alm_samples"]
        phi_s = d["phi_samples"][::thin]
        alm_part = alm_s[::thin, n_lncl:]

        alm_true = d["alm_true_packed"]
        phi_true = d["phi_true_packed"]
        want = packed_length(lmax)
        if alm_true.shape[0] != want:
            raise SystemExit(
                f"{os.path.basename(f)}: field vector is {alm_true.shape[0]} long, "
                f"packed_length({lmax}) is {want} -- this chain predates the "
                "2026-09-06 Im(a_{L,1}) restoration and is a DIFFERENT MODEL "
                "(2L vs 2L+1 dof). Use the packingv2 ensemble, not _doffix."
            )

        for lo, hi in ELL_BINS:
            hi = min(hi, lmax)
            if lo >= hi:
                continue
            for tag, samp, truth in (("phi_power", phi_s, phi_true),
                                     ("alm_power", alm_part, alm_true)):
                post = binned_power(samp, L_arr, lo, hi)
                tru = binned_power(truth, L_arr, lo, hi)
                if post is None:
                    continue
                records.setdefault((tag, lo, hi), []).append(rank_of(tru, post))

    n_draws = alm_part.shape[0] - 1  # rank in {0..n_draws}
    return files, records, n_draws


def plot_rank_histograms(files, records, n_draws, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    for ax, tag, label, color in (
            (axes[0], "phi_power", r"$\phi$ field power", "#4c72b0"),
            (axes[1], "alm_power", "alm field power", "#c44e52")):
        pooled = []
        for lo, hi in ELL_BINS:
            entries = records.get((tag, lo, hi))
            if entries:
                pooled.extend(entries)
        u = (np.array(pooled, dtype=np.float64) + 0.5) / (n_draws + 1.0)
        ax.hist(u, bins=10, range=(0, 1), color=color, alpha=0.75,
                edgecolor="white", density=True)
        ax.axhline(1.0, color="0.3", lw=1.2, ls="--", label="uniform")
        cal_p = discrete_uniform_p(np.array(pooled, dtype=np.int64), n_draws)
        sd_u = rank_spread(np.array(pooled, dtype=np.int64), n_draws)
        ax.set_title(f"{label}\n"
                     rf"$\bar u$={u.mean():.3f}, cal$_p$={cal_p:.3f}, "
                     rf"sd$_u$={sd_u:.3f}", fontsize=10)
        ax.set_xlabel("normalised rank $u$")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("density")
    fig.suptitle(f"SBC rank histograms, pooled over {len(ELL_BINS)} "
                 r"$\ell$-bins, $N$=" + f"{len(files)} chains "
                 f"({n_draws + 1} draws/chain)", y=1.02, fontsize=11)
    fig.text(0.5, -0.04,
             "Block 4 OFF, phi prior proper and identical to the generative "
             "process. cal_p is the simulation-calibrated test (not KS, which "
             "over-rejects on discrete ranks); a bin is flagged at cal_p<0.01. "
             "No bin flagged in this configuration.",
             ha="center", va="top", fontsize=7.5, color="0.25", wrap=True)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"wrote {outpath}")


def plot_power_curve(n_real, n_draws, n_rep, seed, outpath):
    rng = np.random.default_rng(seed)
    null_means = np.array([sbc_ranks(n_real, n_draws, rng).mean()
                           for _ in range(n_rep)])
    lo, hi = np.percentile(null_means, [2.5, 97.5])

    shifts = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    power = np.empty_like(shifts)
    for i, b in enumerate(shifts):
        means = np.array([sbc_ranks(n_real, n_draws, rng, mean_shift=b).mean()
                          for _ in range(n_rep)])
        power[i] = np.mean((means < lo) | (means > hi))

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(shifts, power, "o-", color="#1f4e79", lw=1.8, ms=5)
    ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel(r"injected posterior mean shift ($\sigma$)")
    ax.set_ylabel("P(detected)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"Validation power at $N$={n_real} realizations, "
                 f"{n_draws} draws/chain", fontsize=10)
    fig.text(0.5, -0.02,
             "A 'pass' excludes only shifts above the power curve's knee -- "
             "stated here as a bound, not asserted as exactness.",
             ha="center", va="top", fontsize=7.5, color="0.25", wrap=True)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"wrote {outpath}")
    for b, p in zip(shifts, power):
        print(f"  shift={b:.1f} sigma  power={p:6.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--indir",
        default="results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_packingv2",
        help=("MUST be the restored 2L+1-packing dir (field width "
              "packed_length(lmax)=4092 at lmax=64), not '..._doffix' which "
              "is pre-restoration 2L packing (width 4030) -- see "
              "achievements.md / ROADMAP.md packing-version discipline."))
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--n_rep", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--outdir",
        default="/cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure1")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    files, records, n_draws = collect_ranks(args.indir, args.thin)
    plot_rank_histograms(files, records, n_draws,
                         os.path.join(args.outdir, "rank_histograms.pdf"))
    plot_power_curve(len(files), n_draws + 1, args.n_rep, args.seed,
                     os.path.join(args.outdir, "validation_power.pdf"))


if __name__ == "__main__":
    main()
