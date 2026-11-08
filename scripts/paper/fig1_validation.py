"""Paper figure 1 -- the credibility gate: SBC ranks for all sampled fields, + power.

Every claim after this figure is only as credible as the sampler validated
here, so this figure has to cover the sampler the TITLE claims: a joint Gibbs
sampler over (alm, C_l, phi, C_L^phiphi). THREE standalone panel PDFs:

  (a) rank_histograms.pdf   -- pooled phi and alm field-power ranks over the
                               4 l-bins of every chain (N = 4 x chains). The
                               band treats those N ranks as independent, so it
                               is only a guide: bins of one chain are
                               correlated, and the calibrated test is per bin.
  (b) clpp_sbc.pdf          -- the STRICT SBC rank of C_L^phiphi itself: the
                               rank of the true C_L^phiphi among the Block 4
                               samples that produced it.
  (c) validation_power.pdf  -- P(detect) vs injected posterior-mean shift at
                               this N -- turns "it passes" into a stated bound.
  (d) cltt_sbc.pdf          -- strict ensembles only (--cl_prior_nu, decision
                               D7): the strict SBC rank of C_l^TT, the Block 1
                               twin of (b). There the a_lm row is read against
                               uniform, since its truth comes from the prior.
  (e) rank_moments_by_bin.pdf -- the per-bin test itself (p_bin): rank mean and
                               rank spread per l-bin for every block, against
                               the 95% interval of the same statistic under the
                               null at this many chains. This is where a reader
                               sees WHICH bin fails (the pooled histograms of
                               (a) cannot show it).

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
      --indir results/analysis/ens_exact_l64_A3000_n30 \
      --thin 10 \
      --outdir /cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure3
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/
import _paths  # noqa: E402,F401  (puts every script folder on sys.path)
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
# Written into the ensemble directory by scripts/calibration/null_alm_power_rank_flat_prior.py.
ALM_NULL_FILE = "null_alm_power_rank_flat_prior.npz"


def alm_null_file(thin):
    """Null filename for a thinning: the legacy name at thin 10, else _thin<N>."""
    return ALM_NULL_FILE if thin == 10 else ALM_NULL_FILE.replace(
        ".npz", f"_thin{thin}.npz")


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

    n_draws = alm_part.shape[0]  # draw count; rank lives in {0..n_draws}
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
            n_draws = len(post)
    return np.asarray(us), n_draws, nu


def is_strict(files):
    """True for a --cl_prior_nu ensemble: C_l^TT carries a proper prior and its
    truth is drawn from it, so every block has a strict SBC rank (decision D7)."""
    d0 = np.load(files[0])
    nu = float(d0["cl_prior_nu"]) if "cl_prior_nu" in d0.files else np.nan
    return bool(np.isfinite(nu) and nu > 0.0 and "cl_true" in d0.files)


def collect_cltt_ranks(files, thin):
    """Strict SBC ranks of the drawn C_l^TT truth among the Block 1 samples.

    The C_l^TT twin of collect_clpp_ranks, with the same bin-mean statistic and
    the same layout (bins outermost, chains inner). Only defined for a strict
    ensemble; returns (None, None, nu) otherwise.
    """
    d0 = np.load(files[0])
    nu = float(d0["cl_prior_nu"]) if "cl_prior_nu" in d0.files else np.nan
    if not is_strict(files):
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
            # ln C_l for l = 2..lmax-1 leads alm_samples.
            post = np.exp(d["alm_samples"][::thin, : lmax - 2])[:, ells - 2].mean(axis=1)
            tru = np.asarray(d["cl_true"], dtype=np.float64)[ells].mean()
            rank = int(np.sum(post < tru))
            us.append((rank + 0.5) / (len(post) + 1.0))
            n_draws = len(post)
    return np.asarray(us), n_draws, nu


def load_alm_null(indir, n_draws, mode, lmax, thin=10):
    """{(lo,hi): null u pool} for the a_lm power rank, or None if not computed.

    Under the flat C_l prior with a fixed-spectrum truth the a_lm power rank is
    NOT uniform even for an exact sampler (ROADMAP T0.1c), so the a_lm row is
    read against what an exact sampler gives. The pool must sit on this
    figure's rank grid u = (r+0.5)/(n_draws+1): a null made at another thinning
    has a different draw count, and comparing against it would be wrong
    without any visible sign, so that is refused.
    """
    path = os.path.join(indir, alm_null_file(thin))
    if not os.path.exists(path):
        return None
    d = np.load(path)
    out = {}
    for lo, hi in ELL_BINS:
        hi = min(hi, lmax)
        key = f"{mode}_{lo}_{hi}_u"
        if lo >= hi or key not in d.files:
            continue
        u = np.asarray(d[key], dtype=np.float64)
        r = u * (n_draws + 1.0) - 0.5
        if not (np.allclose(r, np.rint(r), atol=1e-6) and r.max() <= n_draws + 1e-6):
            raise SystemExit(
                f"{path}: '{key}' is not on the rank grid for {n_draws + 1} draws "
                "per chain -- the null was computed at a different thinning. "
                "Re-run null_alm_power_rank_flat_prior.py to match, or pass "
                "--alm_null uniform.")
        out[(lo, hi)] = u
    return out or None


def _sim_u(rng, n, n_draws, pool):
    """n null u values: discrete-uniform, or resampled from an exact-null pool."""
    if pool is None:
        return (rng.integers(0, n_draws + 1, size=n) + 0.5) / (n_draws + 1.0)
    return rng.choice(pool, size=n, replace=True)


def uniform_band(n_samples, n_draws, n_bins=N_HIST_BINS, n_rep=4000, seed=7,
                 null_pools=None):
    """Central 68% and 95% of histogram bin DENSITY under the null.

    Simulated rather than approximated: the rank of a truth among n_draws exact
    draws is discrete-uniform on {0..n_draws}, and this project's standing
    calibration lesson (discrete_uniform_p) is that continuous approximations
    misjudge exactly this statistic.

    `null_pools`, if given, is a list of (pool, n_obs) per ell bin: each
    replicate resamples n_obs values from every bin's exact-null pool and pools
    them, mirroring how the observed histogram pools the bins.
    """
    rng = np.random.default_rng(seed)
    heights = np.empty((n_rep, n_bins))
    for i in range(n_rep):
        if null_pools is None:
            u = _sim_u(rng, n_samples, n_draws, None)
        else:
            u = np.concatenate([_sim_u(rng, n, n_draws, pool) for pool, n in null_pools])
        h, _ = np.histogram(u, bins=n_bins, range=(0, 1), density=True)
        heights[i] = h
    return (np.percentile(heights, [16, 84], axis=0),
            np.percentile(heights, [2.5, 97.5], axis=0))


def _draw_hist(ax, u, n_draws, color, label, null_pools=None):
    b68, b95 = uniform_band(u.size, n_draws, null_pools=null_pools)
    name = "uniform" if null_pools is None else "exact null"
    edges = np.linspace(0, 1, N_HIST_BINS + 1)
    ax.stairs(b95[1], edges, baseline=b95[0], fill=True, color=ps.COL_NULL,
              alpha=0.45, lw=0, label=f"{name}, 95%")
    ax.stairs(b68[1], edges, baseline=b68[0], fill=True, color=ps.COL_NULL,
              alpha=0.75, lw=0, label=f"{name}, 68%")
    ax.hist(u, bins=N_HIST_BINS, range=(0, 1), histtype="step", color=color,
            lw=1.4, density=True, label=label)
    ax.axhline(1.0, color="0.45", lw=0.7, ls="--", zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.margins(y=0.10)  # headroom: a tall first bin must never touch the frame
    ax.set_xlabel("normalised rank $u$")


def mean_sd_p(ranks, n_draws, n_rep=20000, seed=11, null_u=None):
    """Two-sided simulated p-values of the rank MEAN and rank SPREAD.

    The mean is the statistic the power curve (validation_power.pdf) is
    computed for, so it is the one whose p-value the bound refers to. The
    spread (sd_u) is the only summary sensitive to an over-narrow posterior.
    `discrete_uniform_p` (chi-square over n_draws+1 categories) is printed for
    continuity but not drawn: at N=96 over 61 categories it has ~1.6 counts per
    cell and almost no power, so its large p-values say little.

    `null_u`, if given, is an exact-sampler null pool of u values to test
    against instead of the discrete uniform (see load_alm_null).
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(ranks)
    u = (r + 0.5) / (n_draws + 1.0)
    sim = _sim_u(rng, n_rep * r.size, n_draws, null_u).reshape(n_rep, r.size)
    m0, s0 = sim.mean(axis=1), sim.std(axis=1)

    def two_sided(null, obs):
        c = np.median(null)
        return float(np.mean(np.abs(null - c) >= abs(obs - c)))

    return two_sided(m0, u.mean()), two_sided(s0, u.std())


def binned_test(rank_sets, n_draws, null_sets=None):
    """Per-bin mean and spread p-values, Bonferroni-corrected minimum.

    The pooled N=96 ranks are 24 chains x 4 l-bins, and bins from one chain
    are correlated, so a pooled test that assumes 96 independent ranks is
    over-confident -- at thin=60 it "rejects" a_lm spread at p=0.003 where
    a_lm draws are certainly independent. Each bin alone has 24 independent
    chains, so the valid test is per bin, corrected for 4 bins x 2 statistics.
    `null_sets`, if given, is the matching per-bin list of null pools.
    """
    ps_all = []
    null_sets = [None] * len(rank_sets) if null_sets is None else null_sets
    for r, pool in zip(rank_sets, null_sets):
        if len(r) == 0:           # bin above this ensemble's lmax
            continue
        ps_all.extend(mean_sd_p(np.asarray(r), n_draws, null_u=pool))
    return min(1.0, len(ps_all) * min(ps_all)), ps_all


def plot_rank_histograms(records, n_draws, outpath, alm_null=None):
    """alm_null: {(lo,hi): pool} from load_alm_null, or None for uniform."""
    fig, axes = plt.subplots(1, 2, figsize=ps.FIG_2COL, sharey=True)
    for ax, tag, label, color, null in (
            (axes[0], "phi_power", r"$\phi$ field power", ps.COL_PHI, None),
            (axes[1], "alm_power", r"$a_{\ell m}$ field power", ps.COL_ALM, alm_null)):
        bins = records_bins(records, tag)
        pooled = np.asarray([r for b in bins for r in records[(tag,) + b]],
                            dtype=np.int64)
        u = (pooled + 0.5) / (n_draws + 1.0)
        null_sets = None if null is None else [null[b] for b in bins]
        null_pools = (None if null is None else
                      [(null[b], len(records[(tag,) + b])) for b in bins])
        _draw_hist(ax, u, n_draws, color, label, null_pools=null_pools)
        cal_p = discrete_uniform_p(pooled, n_draws)
        sd_u = rank_spread(pooled, n_draws)
        p_corr, p_bins = binned_test([records[(tag,) + b] for b in bins], n_draws,
                                     null_sets=null_sets)
        ps.stat_box(ax, rf"$\bar u$={u.mean():.3f},  sd$_u$={sd_u:.3f}" "\n"
                    rf"$p_{{\rm bin}}$={p_corr:.2f},  $N$={u.size}", loc="lower left")
        print(f"  {tag}: mean_u={u.mean():.4f} sd_u={sd_u:.4f} N={u.size}; per-bin "
              f"(mean,sd) p = {np.round(p_bins, 3).tolist()} -> corrected min "
              f"{p_corr:.3f}  [vs {'uniform' if null is None else 'exact null'}; "
              f"pooled chi2 cal_p={cal_p:.4f}, not drawn]")
    axes[0].set_ylabel("density")
    for ax in axes:
        ax.set_ylim(0, 2.35)       # headroom so the legend clears the bands
    h, lab = axes[0].get_legend_handles_labels()
    axes[0].legend(h[2:], lab[2:], loc="upper right")   # field name only
    h, lab = axes[1].get_legend_handles_labels()
    axes[1].legend(h[2:] + h[:2], lab[2:] + lab[:2], loc="upper right", ncol=3,
                   columnspacing=1.0)
    ps.save(fig, outpath)


def bin_null_interval(n_chains, n_draws, pool=None, alpha=0.05, n_rep=40000, seed=13):
    """Central (1 - alpha) intervals of the rank mean and spread over n_chains under the null."""
    rng = np.random.default_rng(seed)
    sim = _sim_u(rng, n_rep * n_chains, n_draws, pool).reshape(n_rep, n_chains)
    q = [50 * alpha, 100 - 50 * alpha]
    return np.percentile(sim.mean(axis=1), q), np.percentile(sim.std(axis=1), q)


def plot_rank_moments(blocks, n_draws, outpath, alpha=0.05):
    """(e) The per-bin statistics behind p_bin, for every block.

    `blocks` is a list of (label, color, {(lo, hi): ranks}, {(lo, hi): pool or None}).
    Each point is one l-bin of one block: the mean (left) and sd (right) of the
    normalised rank over the chains, with the null's interval at that chain
    count AT THE CORRECTED LEVEL alpha / n_tests, so that a point outside its
    band is a point that fails p_bin (an uncorrected 95% band put passing points
    outside it). A point is also drawn open when its Bonferroni-corrected p
    (over all bins x 2 statistics of that block, as p_bin) falls below alpha.
    """
    fig, axes = plt.subplots(1, 2, figsize=ps.FIG_2COL)
    n_blk = len(blocks)
    width = 0.8 / n_blk
    for k, (label, color, per_bin, pools) in enumerate(blocks):
        # Keys may be truncated at lmax ((30, 40) for (30, 60)): place by lower edge.
        bins = sorted(b for b in per_bin if len(per_bin[b]) > 0)
        n_tests = 2 * len(bins)
        for i, b in enumerate(bins):
            x = [lo for lo, _ in ELL_BINS].index(b[0]) + (k - (n_blk - 1) / 2) * width
            r = np.asarray(per_bin[b])
            u = (r + 0.5) / (n_draws + 1.0)
            p_mean, p_sd = mean_sd_p(r, n_draws, null_u=pools.get(b))
            (m_lo, m_hi), (s_lo, s_hi) = bin_null_interval(len(r), n_draws, pools.get(b),
                                                           alpha=alpha / n_tests)
            for ax, val, lo, hi, pv in ((axes[0], u.mean(), m_lo, m_hi, p_mean),
                                        (axes[1], u.std(), s_lo, s_hi, p_sd)):
                first = k == 0 and i == 0 and ax is axes[0]
                ax.fill_between([x - width / 2, x + width / 2], lo, hi,
                                color=ps.COL_NULL, alpha=0.6, lw=0,
                                label=rf"null, $\alpha/{n_tests}$" if first else None)
                failed = min(1.0, n_tests * pv) < alpha
                ax.plot(x, val, "o", ms=4, color=color,
                        mfc="white" if failed else color,
                        label=label if (i == 0 and ax is axes[0]) else None)
            print(f"  {label:>12s} [{b[0]:2d},{b[1]:2d}): mean_u {u.mean():.3f} "
                  f"(null at alpha/{n_tests} {m_lo:.3f}-{m_hi:.3f}, p {p_mean:.3f}); sd_u {u.std():.3f} "
                  f"(null {s_lo:.3f}-{s_hi:.3f}, p {p_sd:.3f})")
    for ax, ylab, ref in ((axes[0], r"rank mean $\bar u$ per bin", 0.5),
                          (axes[1], r"rank spread sd$_u$ per bin", 1 / np.sqrt(12))):
        ax.axhline(ref, color="0.45", lw=0.7, ls="--", zorder=0)
        ax.set_xticks(range(len(ELL_BINS)))
        ax.set_xticklabels([f"[{lo},{hi})" for lo, hi in ELL_BINS])
        ax.set_xlabel(r"multipole bin, $\ell$ or $L$")
        ax.set_ylabel(ylab)
    axes[0].legend(loc="upper left", ncol=2)
    n_ch = len(next(iter(blocks[0][2].values())))
    ps.stat_box(axes[0], rf"$N$={n_ch} chains per bin", loc="lower right")
    ps.save(fig, outpath)


def records_bins(records, tag):
    """The (lo, hi) ell bins present for `tag`, in ELL_BINS order."""
    return [(lo, hi) for (t, lo, hi) in sorted(records, key=lambda k: k[1]) if t == tag]


def plot_clpp_sbc(us, n_draws, nu, n_chains, outpath, tag="clpp",
                  label=r"$C_L^{\phi\phi}$ (strict SBC)", color=None):
    # Same calibrated statistics as the field panels. The continuous KS test
    # this panel used to print is the miscalibrated one (achievements.md,
    # 2026-09-12) -- the paper must not carry two different tests side by side.
    ranks = np.rint(us * (n_draws + 1.0) - 0.5).astype(np.int64)
    cal_p = discrete_uniform_p(ranks, n_draws)
    sd_u = rank_spread(ranks, n_draws)
    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    _draw_hist(ax, us, n_draws, ps.COL_CLPP if color is None else color, label)
    ax.set_ylim(0, 2.35)
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h[2:], lab[2:], loc="upper right")
    # collect_clpp_ranks loops bins outermost, chains inner; bins above the
    # ensemble's lmax are skipped there, so split by chain count, not by
    # len(ELL_BINS).
    p_corr, p_bins = binned_test(ranks.reshape(-1, n_chains), n_draws)
    ps.stat_box(ax, rf"$\bar u$={us.mean():.3f},  sd$_u$={sd_u:.3f}" "\n"
                rf"$p_{{\rm bin}}$={p_corr:.2f},  $N$={us.size}", loc="lower left")
    ax.set_ylabel("density")
    ps.save(fig, outpath)
    print(f"  {tag} strict SBC (nu={nu:g}): mean_u={us.mean():.4f} sd_u={sd_u:.4f} "
          f"N={us.size}; per-bin (mean,sd) p = {np.round(p_bins, 3).tolist()} -> "
          f"corrected min {p_corr:.3f}  [pooled chi2 cal_p={cal_p:.4f}, not drawn]")


def _two_sided_limits(null, alpha):
    return np.percentile(null, [50 * alpha, 100 - 50 * alpha])


def plot_power_curve(n_real, n_draws, n_rep, seed, outpath, n_tests=8):
    """P(detect) against an injected mean shift, AT THE LEVEL p_bin USES.

    p_bin is the Bonferroni-corrected minimum over n_tests per-bin tests (4
    l-bins x {rank mean, rank spread}), so a single bin's mean test only flags
    at alpha / n_tests. Until 2026-11-04 this curve used the uncorrected
    alpha = 0.05, which overstated the power of the test the paper applies
    (50 % power at 0.28 sigma uncorrected). Both curves are now computed; the
    corrected one is the bound, the uncorrected one is drawn dotted for
    reference. The rank SPREAD's power against a too-narrow or too-wide
    posterior (sd_scale) is computed the same way and goes into the table:
    the mean test has no power against a width error.
    """
    rng = np.random.default_rng(seed)
    alpha, alpha_c = 0.05, 0.05 / n_tests
    n_null = 10 * n_rep
    null = np.array([sbc_ranks(n_real, n_draws, rng) for _ in range(n_null)])
    m_lim = {a: _two_sided_limits(null.mean(axis=1), a) for a in (alpha, alpha_c)}
    s_lim = {a: _two_sided_limits(null.std(axis=1), a) for a in (alpha, alpha_c)}

    def power(stat, lim):
        return float(np.mean((stat < lim[0]) | (stat > lim[1])))

    shifts = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    p_mean = np.empty((shifts.size, 2))
    for i, b in enumerate(shifts):
        means = np.array([sbc_ranks(n_real, n_draws, rng, mean_shift=b).mean()
                          for _ in range(n_rep)])
        p_mean[i] = power(means, m_lim[alpha_c]), power(means, m_lim[alpha])
    scales = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5])
    p_sd = np.empty((scales.size, 2))
    for i, c in enumerate(scales):
        sds = np.array([sbc_ranks(n_real, n_draws, rng, sd_scale=c).std()
                        for _ in range(n_rep)])
        p_sd[i] = power(sds, s_lim[alpha_c]), power(sds, s_lim[alpha])

    # Where the corrected curve crosses 50% -- the number the caption quotes.
    knee = float(np.interp(0.5, p_mean[:, 0], shifts))
    knee_single = float(np.interp(0.5, p_mean[:, 1], shifts))

    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    ax.plot(shifts, p_mean[:, 0], "o-", color=ps.COL_AWARE, ms=3,
            label=rf"per-bin test at $\alpha/{n_tests}$ (as $p_{{\rm bin}}$)")
    ax.plot(shifts, p_mean[:, 1], "o:", color=ps.COL_AWARE, ms=2.5, alpha=0.6,
            label=r"one bin alone, $\alpha=0.05$")
    ax.axhline(0.5, color=ps.COL_NULL, lw=0.8, ls=":")
    ax.axvline(knee, color=ps.COL_NULL, lw=0.8, ls=":")
    ax.set_xlabel(r"injected posterior mean shift ($\sigma$)")
    ax.set_ylabel("P(detected)")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="upper left")
    ps.stat_box(ax, rf"$N$={n_real} chains, {n_draws} draws" "\n"
                rf"50% power at {knee:.2f}$\sigma$", loc="lower right")
    ps.save(fig, outpath)
    for b, (pc, p1) in zip(shifts, p_mean):
        print(f"  shift={b:.1f} sigma  power={pc:6.1%}  (single test {p1:6.1%})")
    for c, (pc, p1) in zip(scales, p_sd):
        print(f"  sd_scale={c:.2f}  spread power={pc:6.1%}  (single test {p1:6.1%})")
    print(f"  50% power at {knee:.3f} sigma at alpha/{n_tests} "
          f"({knee_single:.3f} sigma for one uncorrected test)")
    write_power_table(shifts, p_mean, scales, p_sd, knee, n_real, n_draws, n_tests,
                      os.path.splitext(outpath)[0] + "_table")


def write_power_table(shifts, p_mean, scales, p_sd, knee, n_real, n_draws, n_tests, stem):
    """<stem>.csv and <stem>.tex: the power curve as the paper's table.

    Written by the same run as the curve, so the table and the figure can
    never disagree. Columns: power at alpha/n_tests (the test p_bin applies),
    then for one uncorrected test. The tex adds the spread's power against a
    posterior whose width is wrong by sd_scale.
    """
    np.savetxt(stem + ".csv", np.column_stack([shifts, p_mean]), delimiter=",",
               fmt=("%.2f", "%.4f", "%.4f"),
               header="shift_sigma,power_corrected,power_single", comments="")
    rows = "\n".join(rf"{b:.1f} & {100 * p[0]:.1f}\% & {100 * p[1]:.1f}\% \\"
                     for b, p in zip(shifts, p_mean))
    rows_sd = "\n".join(rf"{c:.2f} & {100 * p[0]:.1f}\% & {100 * p[1]:.1f}\% \\"
                        for c, p in zip(scales, p_sd))
    with open(stem + ".tex", "w") as fh:
        fh.write("% generated by scripts/paper/fig1_validation.py; do not edit\n"
                 f"% {n_real} chains, {n_draws} draws per chain; 50% power at {knee:.3f} sigma"
                 f" (alpha/{n_tests})\n"
                 "\\begin{tabular}{ccc}\n\\hline\n"
                 rf"mean shift ($\sigma$) & $P$, $\alpha/{n_tests}$ & $P$, one bin \\" "\n\\hline\n"
                 f"{rows}\n\\hline\n"
                 rf"width factor & $P$, $\alpha/{n_tests}$ & $P$, one bin \\" "\n\\hline\n"
                 f"{rows_sd}\n\\hline\n\\end{{tabular}}\n")
    print(f"wrote {stem}.csv, {stem}.tex")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--indir",
        default="results/analysis/ens_exact_l64_A3000_n30",
        help=("MUST be the restored 2L+1-packing dir (field width "
              "packed_length(lmax)=4092 at lmax=64), not '..._doffix' which is "
              "pre-restoration 2L packing -- see achievements.md."))
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--n_rep", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--alm_null", choices=("effective", "nominal", "uniform"),
        default="effective",
        help=("What the a_lm row is read against. Default: the exact-sampler "
              f"null in <indir>/{ALM_NULL_FILE} at chain-calibrated effective "
              "noise (ROADMAP T0.1c). Falls back to uniform, with a notice, if "
              "that file is absent."))
    ap.add_argument(
        "--outdir",
        # paper renumbered 2026-10-26 (was figure1)
        default="/cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure3")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    files = chain_files(args.indir)
    print(f"{len(files)} chains from {args.indir}, thin={args.thin}")

    records, n_draws = collect_field_ranks(files, args.thin)
    alm_null = None
    strict = is_strict(files)
    if strict:
        print("  strict ensemble (cl_prior_nu set): the a_lm truth is drawn from the "
              "sampler's own prior, so its rank is SBC-supported and read against "
              "uniform; a flat-prior null does not apply (D7)")
    elif args.alm_null != "uniform":
        lmax = int(np.load(files[0])["lmax"])
        alm_null = load_alm_null(args.indir, n_draws, args.alm_null, lmax,
                                 thin=args.thin)
        if alm_null is None:
            print(f"  ! no {alm_null_file(args.thin)} in {args.indir}: the a_lm row is read "
                  "against UNIFORM, which is non-uniform by construction under "
                  "the flat C_l prior -- run null_alm_power_rank_flat_prior.py.")
        else:
            print(f"  a_lm row read against the {args.alm_null} exact-sampler null")
    plot_rank_histograms(records, n_draws,
                         os.path.join(args.outdir, "rank_histograms.pdf"),
                         alm_null=alm_null)

    us, clpp_draws, nu = collect_clpp_ranks(files, args.thin)
    if us is None:
        print("  ! Block 4 OFF or improper prior in this ensemble -- the strict "
              "C_L^phiphi rank is NOT a valid SBC statistic here, so panel (b) "
              "is skipped rather than drawn misleadingly.")
    else:
        plot_clpp_sbc(us, clpp_draws, nu, len(files),
                      os.path.join(args.outdir, "clpp_sbc.pdf"))

    if strict:
        us_tt, tt_draws, nu_tt = collect_cltt_ranks(files, args.thin)
        plot_clpp_sbc(us_tt, tt_draws, nu_tt, len(files),
                      os.path.join(args.outdir, "cltt_sbc.pdf"), tag="cltt",
                      label=r"$C_\ell^{TT}$ (strict SBC)", color=ps.COL_CLTT)

    def split_bins(us_blk, n_dr):
        """Bins-outermost u vector -> {(lo, hi): integer ranks}."""
        ranks = np.rint(us_blk * (n_dr + 1.0) - 0.5).astype(np.int64)
        present = [b for b in ELL_BINS if b[0] < min(b[1], lmax_ens)]
        return dict(zip(present, ranks.reshape(-1, len(files))))

    lmax_ens = int(np.load(files[0])["lmax"])
    blocks = []
    if strict:
        blocks.append((r"$C_\ell^{TT}$", ps.COL_CLTT, split_bins(us_tt, tt_draws), {}))
    alm_pools = {} if alm_null is None else dict(alm_null)
    blocks.append((r"$a_{\ell m}$", ps.COL_ALM,
                   {(lo, hi): records[("alm_power", lo, hi)]
                    for (_t, lo, hi) in records if _t == "alm_power"}, alm_pools))
    blocks.append((r"$\phi$", ps.COL_PHI,
                   {(lo, hi): records[("phi_power", lo, hi)]
                    for (_t, lo, hi) in records if _t == "phi_power"}, {}))
    if us is not None:
        blocks.append((r"$C_L^{\phi\phi}$", ps.COL_CLPP, split_bins(us, clpp_draws), {}))
    if len({n_draws, clpp_draws} - {None}) == 1 and (not strict or tt_draws == n_draws):
        plot_rank_moments(blocks, n_draws,
                          os.path.join(args.outdir, "rank_moments_by_bin.pdf"))
    else:
        print("  ! blocks have different draw counts; rank_moments_by_bin.pdf skipped")

    n_tests = 2 * len([b for b in ELL_BINS if b[0] < min(b[1], lmax_ens)])
    plot_power_curve(len(files), n_draws, args.n_rep, args.seed,
                     os.path.join(args.outdir, "validation_power.pdf"), n_tests=n_tests)


if __name__ == "__main__":
    main()
