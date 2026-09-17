"""Paper figure 1 -- the credibility gate: SBC ranks for all sampled fields, + power.

Every claim after this figure is only as credible as the sampler validated
here, so this figure has to cover the sampler the TITLE claims: a joint Gibbs
sampler over (alm, C_l, phi, C_L^phiphi). THREE standalone panel PDFs:

  (a) rank_histograms.pdf   -- pooled phi and alm field-power ranks, N=24
                               chains at thin=10 (~60 draws/chain).
  (b) clpp_sbc.pdf          -- the STRICT SBC rank of C_L^phiphi itself: the
                               rank of the true C_L^phiphi among the Block 4
                               samples that produced it.
  (c) validation_power.pdf  -- P(detect) vs injected posterior-mean shift at
                               this N -- turns "it passes" into a stated bound.

WHY PANEL (b) EXISTS. Until 2026-09-18 this figure showed only the phi and alm
field ranks. Those certify two of the four blocks. The paper's title claims the
fourth -- C_L^phiphi -- and a referee who notices that the validation figure
never ranks it has found a real hole, not a presentational one. The strict rank
is the right statistic to close it: it needs no null simulation, because with a
PROPER prior (--cl_phiphi_prior_nu) the truth is drawn from the very prior the
sampler targets, so the rank is uniform under a correct sampler by the SBC
theorem. It is also the statistic that caught the 2026-08-31 inverse-Gamma dof
bug, so it has demonstrated power against exactly the failure mode that matters.
It is computed by importing validate_coverage_rank_nulls.strict_clpp_sbc's own
logic, not by reimplementing it.

WHY THE UNIFORM BAND IN (a) AND (b) IS NOT DECORATION. A rank histogram with a
bare dashed line at 1.0 cannot be read: the eye sees a tall first bin and has no
way to judge whether it is a 1-sigma wobble or a detection. The shaded band is
the central 68/95% of bin heights under the exact discrete-uniform null at this
N and this many draws -- simulated, not a Gaussian approximation, because the
rank is discrete-uniform and the whole project's calibration discipline
(discrete_uniform_p's docstring) is that continuous approximations over-reject
here. With the band drawn, "no bin flagged" is something the reader verifies
rather than takes on trust.

THE ENSEMBLE. Defaults to ..._properprior_packingv2: N=24, restored 2L+1
packing, Block 4 ON, proper prior nu=6 -- the ensemble achievements.md names as
certifying the title's object. The script asserts the field width against
packed_length(lmax) and refuses to run on a pre-2026-09-06 (2L) directory,
which is a different model. NOTE: a previous version of this figure carried a
footer reading "Block 4 OFF" while loading exactly this Block-4-ON ensemble --
a false statement, baked into a picture where no test could reach it. That is
why paper_style forbids prose inside panels; all such text now lives in the
LaTeX caption.

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/paper/fig1_validation.py \
      --indir results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_packingv2 \
      --thin 10 \
      --outdir /cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure1
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paper_style as ps  # noqa: E402

ps.apply()  # installs Agg + paper rcParams before pyplot is touched
import matplotlib.pyplot as plt  # noqa: E402
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
N_HIST_BINS = 10


def chain_files(indir):
    files = sorted(f for f in glob.glob(os.path.join(indir, "chain_r*.npz"))
                   if not f.endswith("_ckpt.npz"))
    if not files:
        raise SystemExit(f"no chains in {indir}")
    return files


def collect_field_ranks(files, thin):
    """Per-bin ranks for phi_power and alm_power, mirroring aggregate_coverage_ranks."""
    records = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        lmax = int(d["lmax"])
        n_lncl = lmax - 2
        n_real = lmax * (lmax + 1) // 2 - 3
        n_imag = packed_sizes(lmax)[1]
        L_arr, _m = _alm_index_lm(lmax, n_real, n_imag)

        phi_s = d["phi_samples"][::thin]
        alm_part = d["alm_samples"][::thin, n_lncl:]

        alm_true = d["alm_true_packed"]
        phi_true = d["phi_true_packed"]
        want = packed_length(lmax)
        if alm_true.shape[0] != want:
            raise SystemExit(
                f"{os.path.basename(f)}: field vector is {alm_true.shape[0]} long, "
                f"packed_length({lmax}) is {want} -- this chain predates the "
                "2026-09-06 Im(a_{L,1}) restoration and is a DIFFERENT MODEL "
                "(2L vs 2L+1 dof). Use the packingv2 ensemble.")

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

    n_draws = alm_part.shape[0] - 1  # rank lives in {0..n_draws}
    return records, n_draws


def collect_clpp_ranks(files, thin):
    """Strict SBC ranks of C_L^phiphi_true among the Block 4 samples.

    Same computation as validate_coverage_rank_nulls.strict_clpp_sbc (which
    prints rather than returns); the rank expression is lifted from it
    verbatim so the panel and that script cannot drift apart.

    Returns (us, n_draws) or (None, None) if this ensemble has no Block 4 or
    no proper prior -- in which case the rank is not a valid SBC statistic and
    the panel must not be drawn.
    """
    d0 = np.load(files[0])
    nu = float(d0["cl_phiphi_prior_nu"]) if "cl_phiphi_prior_nu" in d0.files else 0.0
    if "cl_phiphi_samples" not in d0.files or not (np.isfinite(nu) and nu > 0.0):
        return None, None, nu

    us = []
    n_draws = None
    for lo, hi in ELL_BINS:
        for f in files:
            d = np.load(f)
            lmax = int(d["lmax"])
            ells = np.arange(max(2, lo), min(lmax, hi))
            if ells.size == 0:
                continue
            # cl_phiphi_samples is stored as log C_L for l = 2..lmax-1.
            post = np.exp(d["cl_phiphi_samples"][::thin])[:, ells - 2].mean(axis=1)
            tru = np.asarray(d["cl_phiphi_true"], dtype=np.float64)[ells].mean()
            rank = int(np.sum(post < tru))
            us.append((rank + 0.5) / (len(post) + 1.0))
            n_draws = len(post) - 1
    return np.asarray(us), n_draws, nu


def uniform_band(n_samples, n_draws, n_bins=N_HIST_BINS, n_rep=4000, seed=7):
    """Central 68% and 95% of histogram bin DENSITY under the discrete-uniform null.

    Simulated rather than approximated: the rank of a truth among n_draws exact
    draws is discrete-uniform on {0..n_draws}, and this project's standing
    calibration lesson (discrete_uniform_p) is that continuous approximations
    misjudge exactly this statistic.
    """
    rng = np.random.default_rng(seed)
    heights = np.empty((n_rep, n_bins))
    for i in range(n_rep):
        r = rng.integers(0, n_draws + 1, size=n_samples)
        u = (r + 0.5) / (n_draws + 1.0)
        h, _ = np.histogram(u, bins=n_bins, range=(0, 1), density=True)
        heights[i] = h
    return (np.percentile(heights, [16, 84], axis=0),
            np.percentile(heights, [2.5, 97.5], axis=0))


def _draw_hist(ax, u, n_draws, color, label):
    b68, b95 = uniform_band(u.size, n_draws)
    edges = np.linspace(0, 1, N_HIST_BINS + 1)
    ax.stairs(b95[1], edges, baseline=b95[0], fill=True, color=ps.COL_NULL,
              alpha=0.45, lw=0, label="uniform, 95%")
    ax.stairs(b68[1], edges, baseline=b68[0], fill=True, color=ps.COL_NULL,
              alpha=0.75, lw=0, label="uniform, 68%")
    ax.hist(u, bins=N_HIST_BINS, range=(0, 1), histtype="step", color=color,
            lw=1.4, density=True, label=label)
    ax.axhline(1.0, color="0.45", lw=0.7, ls="--", zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.margins(y=0.10)  # headroom: a tall first bin must never touch the frame
    ax.set_xlabel("normalised rank $u$")


def plot_rank_histograms(records, n_draws, outpath):
    fig, axes = plt.subplots(1, 2, figsize=ps.FIG_2COL, sharey=True)
    for ax, tag, label, color in (
            (axes[0], "phi_power", r"$\phi$ field power", ps.COL_PHI),
            (axes[1], "alm_power", r"$a_{\ell m}$ field power", ps.COL_ALM)):
        pooled = []
        for lo, hi in ELL_BINS:
            pooled.extend(records.get((tag, lo, hi), []))
        pooled = np.asarray(pooled, dtype=np.int64)
        u = (pooled + 0.5) / (n_draws + 1.0)
        _draw_hist(ax, u, n_draws, color, label)
        cal_p = discrete_uniform_p(pooled, n_draws)
        sd_u = rank_spread(pooled, n_draws)
        ps.stat_box(ax, f"{label}\n" rf"$\bar u$={u.mean():.3f},  cal$_p$={cal_p:.2f}"
                    "\n" rf"sd$_u$={sd_u:.3f},  $N$={u.size}", loc="lower left")
        print(f"  {tag}: mean_u={u.mean():.4f} cal_p={cal_p:.4f} "
              f"sd_u={sd_u:.4f} N={u.size}")
    axes[0].set_ylabel("density")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    ps.save(fig, outpath)


def plot_clpp_sbc(us, n_draws, nu, outpath):
    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    _draw_hist(ax, us, n_draws, ps.COL_CLPP, r"$C_L^{\phi\phi}$")
    from scipy import stats
    ks = stats.kstest(us, "uniform").pvalue
    ps.stat_box(ax, r"$C_L^{\phi\phi}$ strict SBC" "\n"
                rf"$\bar u$={us.mean():.3f},  KS$_p$={ks:.2f}" "\n"
                rf"$\nu$={nu:g},  $N$={us.size}", loc="lower left")
    ax.set_ylabel("density")
    fig.tight_layout()
    ps.save(fig, outpath)
    print(f"  clpp strict SBC: mean_u={us.mean():.4f} KS_p={ks:.4f} N={us.size}")


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

    # Where the curve crosses 50% -- the number the caption quotes as the bound.
    knee = np.interp(0.5, power, shifts)

    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    ax.plot(shifts, power, "o-", color=ps.COL_AWARE, ms=3)
    ax.axhline(0.5, color=ps.COL_NULL, lw=0.8, ls=":")
    ax.axvline(knee, color=ps.COL_NULL, lw=0.8, ls=":")
    ax.set_xlabel(r"injected posterior mean shift ($\sigma$)")
    ax.set_ylabel("P(detected)")
    ax.set_ylim(-0.03, 1.03)
    ps.stat_box(ax, rf"$N$={n_real} chains, {n_draws} draws" "\n"
                rf"50% power at {knee:.2f}$\sigma$", loc="lower right")
    fig.tight_layout()
    ps.save(fig, outpath)
    for b, p in zip(shifts, power):
        print(f"  shift={b:.1f} sigma  power={p:6.1%}")
    print(f"  50% power at {knee:.3f} sigma")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--indir",
        default="results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_packingv2",
        help=("MUST be the restored 2L+1-packing dir (field width "
              "packed_length(lmax)=4092 at lmax=64), not '..._doffix' which is "
              "pre-restoration 2L packing -- see achievements.md."))
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--n_rep", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--outdir",
        default="/cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure1")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    files = chain_files(args.indir)
    print(f"{len(files)} chains from {args.indir}, thin={args.thin}")

    records, n_draws = collect_field_ranks(files, args.thin)
    plot_rank_histograms(records, n_draws,
                         os.path.join(args.outdir, "rank_histograms.pdf"))

    us, clpp_draws, nu = collect_clpp_ranks(files, args.thin)
    if us is None:
        print("  ! Block 4 OFF or improper prior in this ensemble -- the strict "
              "C_L^phiphi rank is NOT a valid SBC statistic here, so panel (b) "
              "is skipped rather than drawn misleadingly.")
    else:
        plot_clpp_sbc(us, clpp_draws, nu,
                      os.path.join(args.outdir, "clpp_sbc.pdf"))

    plot_power_curve(len(files), n_draws + 1, args.n_rep, args.seed,
                     os.path.join(args.outdir, "validation_power.pdf"))


if __name__ == "__main__":
    main()
