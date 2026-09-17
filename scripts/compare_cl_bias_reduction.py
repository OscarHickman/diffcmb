"""C_l^TT bias reduction: lensing-aware joint sampler vs a lensing-blind analysis.

ROADMAP.md section 2 differentiator figure. Both chains are run on the IDENTICAL
simulation (seed=0, lmax=128, nside=128, noisesig=1.0):

  * lensing-blind  -- Phase 0 Gibbs, alm + C_l blocks only, NO phi block. Fits an
    UNLENSED model to LENSED data, so its recovered C_l^TT should absorb the
    lensing smoothing as spurious power -- the Commander-style baseline.
  * lensing-aware  -- the full joint Gibbs (alm, C_l, phi, C_L^phiphi), which
    marginalises over phi and should recover the UNLENSED truth.

The claim being measured is that the lensing-aware posterior sits closer to the
unlensed truth than the blind one does.

Truth here is the REALIZED power of this particular truth sky,
S_l(alm_true)/k_l, NOT the fiducial `cl_true`. Both chains analyse ONE sky, and
a single realization scatters around the fiducial spectrum by cosmic variance --
a factor of several at l=2. Differencing against the fiducial therefore measures
cosmic variance plus bias and calls the sum "bias"; at low l the cosmic-variance
term dominates completely. This is the same trap recorded in achievements.md for
the coverage statistic, and `realized_spectrum` is reused here for exactly that
reason. A second correction on top of that: an unbiased C_l posterior does NOT centre
on the realized spectrum either. Block 1 draws C_l ~ InvGamma(k_l/2 - 1, S_l/2)
under the flat improper prior, whose MEAN is S_l/(k_l - 4) while the realized
power is S_l/k_l -- so a perfectly correct sampler sits high by k_l/(k_l - 4),
which is +15% at l~20 and +2% at l~110. That offset is large, l-dependent, and
common to BOTH chains, so differencing against the realized spectrum directly
measures the prior artifact and buries the lensing signal underneath it (it
reports the lensing-aware chain as the more biased of the two, which is an
artifact of the statistic, not a result).

The reference used here is therefore the EXPECTED posterior mean under a
correct unlensed model, E[C_l] = S_l(alm_true) / (k_l - 4), bin-averaged. A
sampler that has correctly marginalised the lensing away should land on it; a
lensing-blind fit to lensed data should fall BELOW it by a deficit that grows
with l, because lensing smooths the acoustic peaks and an unlensed model
absorbs that as lost small-scale power.

BOTH files must be packed at the CURRENT width -- a pre-2026-09-06 .npz carries
2L dof per multipole and is a different model (achievements.md). A saved .npz
has no PACKING_VERSION, so this script checks the widths itself and refuses to
run on a mismatch rather than silently rendering a packing artifact as physics.

Low-l caveat: the lensing-aware chain's phi block does not equilibrate in
[2,10) at lmax=128 (job 11966631, worst lag-1 0.967), so that bin carries very
few effective samples. It is computed and plotted but marked UNRELIABLE, and
excluded from the headline numbers.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "diffcmb"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_coverage_ranks import _sl_from_packed, realized_spectrum  # noqa: E402

from diffcmb.alm_utils import packed_dof_per_multipole, packed_length  # noqa: E402

# Bins follow the equilibration diagnostics in job 11966631 so the reliability
# verdict per bin maps one-to-one onto a bin here.
BINS_128 = [(2, 10), (10, 30), (30, 60), (60, 100), (100, 128)]


def bins_for(lmax):
    """lmax=128 keeps its original bins exactly (so recorded results reproduce);
    higher lmax appends 32-wide bins above 128, truncated at lmax."""
    if lmax <= 128:
        return [(lo, min(hi, lmax)) for lo, hi in BINS_128 if lo < lmax]
    extra, lo = [], 128
    while lo < lmax:
        extra.append((lo, min(lo + 32, lmax)))
        lo += 32
    return BINS_128 + extra


def unreliable_for(lmax):
    """Bins where the phi block has not equilibrated, so the C_l posterior there
    carries very few effective samples. lmax=128: [2,10) only (job 11966631,
    worst lag-1 0.967). lmax=192: the mode extends up to [10,30) as well
    (job 11987444 gate NO-GO, lag-1 0.928, decorrelation lag 200) -- see
    ROADMAP.md Next action 1. Keyed off lmax rather than hardcoded so a harvest
    at one scale cannot silently inherit another scale's reliability verdict."""
    if lmax <= 128:
        return {(2, 10)}
    return {(2, 10), (10, 30)}


def _blocked_sem(x, n_blocks=20):
    """Standard error of the mean via blocking -- honest under autocorrelation.

    A plain std/sqrt(N) would badly understate the error on a chain with
    tau_int in the tens, which both of these have at low l.
    """
    x = np.asarray(x, dtype=float)
    n = len(x) // n_blocks * n_blocks
    if n < n_blocks:
        return float(np.std(x, ddof=1) / max(np.sqrt(len(x)), 1.0))
    means = x[:n].reshape(n_blocks, -1).mean(axis=1)
    return float(np.std(means, ddof=1) / np.sqrt(n_blocks))


def load_chains(blind_path, aware_path, lmax):
    blind = np.load(blind_path)
    aware = np.load(aware_path)

    want = packed_length(lmax)
    for name, d in (("lensing-blind", blind), ("lensing-aware", aware)):
        got = d["alm_true_packed"].shape[0]
        if got != want:
            raise SystemExit(
                f"PACKING MISMATCH in the {name} chain: alm_true_packed is {got} "
                f"long but packed_length({lmax}) is {want}. This file predates the "
                f"2026-09-06 Im(a_{{L,1}}) restoration and is a DIFFERENT MODEL; "
                f"differencing it would render a packing artifact as physics. "
                f"Re-run it under the current packing."
            )

    n_cl = lmax - 2
    # Blind chain saves C_l directly; the joint chain carries log C_l as the
    # leading block of its packed parameter vector.
    cl_blind = np.asarray(blind["cl_samples"], dtype=float)
    cl_aware = np.exp(np.asarray(aware["alm_samples"][:, :n_cl], dtype=float))

    if not np.allclose(np.asarray(blind["cl_true"], dtype=float),
                       np.asarray(aware["cl_true"], dtype=float)):
        raise SystemExit(
            "The two chains do not share a fiducial spectrum -- they are not the "
            "same simulation, so the comparison is meaningless."
        )
    # The two truth vectors are the same sky in the same packing, so either
    # gives the same realized spectrum; check that rather than assume it.
    a_blind = np.asarray(blind["alm_true_packed"], dtype=float)
    a_aware = np.asarray(aware["alm_true_packed"], dtype=float)
    if not np.allclose(a_blind, a_aware):
        raise SystemExit(
            "The two chains' truth alm differ -- they are not the same sky "
            "realization, so a bias comparison is meaningless."
        )
    cl_realized = realized_spectrum(_sl_from_packed(a_blind, lmax), lmax)
    # Expected posterior mean under a CORRECT unlensed model with the flat
    # improper prior: InvGamma(k/2-1, S/2) has mean S/(k-4) = (S/k)*k/(k-4).
    k = packed_dof_per_multipole(lmax).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(k > 4, k / (k - 4), np.nan)
    cl_expected = cl_realized * factor
    return cl_blind, cl_aware, cl_realized[2:lmax], cl_expected[2:lmax]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blind", default="results/analysis/lensing_blind_baseline_lmax128_packingv2.npz")
    ap.add_argument("--aware", default="results/analysis/pilot_coverage_lmax128_postfix_hmc.npz")
    ap.add_argument("--lmax", type=int, default=128)
    ap.add_argument("--out", default="results/analysis/cl_bias_reduction_lmax128")
    ap.add_argument("--exclude_unreliable", action="store_true",
                    help="omit bins where phi is not equilibrated from the FIGURE "
                         "entirely, instead of shading them. They dominate the "
                         "y-scale (the [2,10) bars are ~20x the others), which "
                         "hides the trend the figure exists to show. The printed "
                         "table always reports every bin.")
    args = ap.parse_args()
    unreliable = unreliable_for(args.lmax)

    cl_blind, cl_aware, cl_realized, cl_expected = load_chains(
        args.blind, args.aware, args.lmax)
    ell = np.arange(2, args.lmax)
    print(f"lensing-blind chain: {cl_blind.shape[0]} draws")
    print(f"lensing-aware chain: {cl_aware.shape[0]} draws")
    print(f"packing verified at packed_length({args.lmax}) = {packed_length(args.lmax)}\n")

    rows = []
    print(f"{'bin':>12} {'blind/exp':>11} {'aware/exp':>11} "
          f"{'blind pull':>11} {'aware pull':>11}   verdict")
    for lo, hi in bins_for(args.lmax):
        m = (ell >= lo) & (ell < hi)
        # Reference = expected posterior mean under a correct unlensed model,
        # NOT the realized power (see module docstring).
        t = cl_expected[m].mean()
        b_per_sweep = cl_blind[:, m].mean(axis=1)
        a_per_sweep = cl_aware[:, m].mean(axis=1)
        b, a = b_per_sweep.mean(), a_per_sweep.mean()
        b_sem, a_sem = _blocked_sem(b_per_sweep), _blocked_sem(a_per_sweep)
        # "Pull": how many of its OWN standard errors each posterior sits from
        # the truth. This is the bias statement; the ratio alone hides whether
        # an offset is significant.
        b_pull, a_pull = (b - t) / b_sem, (a - t) / a_sem
        tag = "UNRELIABLE (phi not equilibrated)" if (lo, hi) in unreliable else ""
        print(f"  [{lo:>3},{hi:>4}) {b/t:>11.4f} {a/t:>11.4f} "
              f"{b_pull:>11.2f} {a_pull:>11.2f}   {tag}")
        rows.append((lo, hi, t, b, a, b_sem, a_sem, b_pull, a_pull))

    print("\n  blind/exp and aware/exp are each posterior's bin-mean divided by the\n"
          "  expected posterior mean of a CORRECT unlensed model. 1.0 = unbiased.\n"
          "  A lensing-blind fit to lensed data should fall BELOW 1, further as l grows.")
    rel = [r for r in rows if (r[0], r[1]) not in unreliable]
    mb = float(np.mean([abs(r[3] / r[2] - 1.0) for r in rel]))
    ma = float(np.mean([abs(r[4] / r[2] - 1.0) for r in rel]))
    print(f"\nMean |fractional bias| over the reliable bins "
          f"({', '.join(f'[{r[0]},{r[1]})' for r in rel)}):")
    print(f"  lensing-blind : {mb:.4f}")
    print(f"  lensing-aware : {ma:.4f}")
    if ma < mb:
        print(f"  -> lensing-aware reduces the bias by {100*(1-ma/mb):.1f}%")
    else:
        print("  -> NO bias reduction measured; do not build the figure from this.")

    np.savez(args.out + ".npz", bins=np.array([(r[0], r[1]) for r in rows]),
             truth=np.array([r[2] for r in rows]),
             blind_mean=np.array([r[3] for r in rows]),
             aware_mean=np.array([r[4] for r in rows]),
             blind_sem=np.array([r[5] for r in rows]),
             aware_sem=np.array([r[6] for r in rows]),
             blind_pull=np.array([r[7] for r in rows]),
             aware_pull=np.array([r[8] for r in rows]),
             cl_realized=cl_realized, cl_expected=cl_expected, ell=ell)
    print(f"\nSaved {args.out}.npz")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; numbers saved, figure skipped.")
        return

    plot_rows = ([r for r in rows if (r[0], r[1]) not in unreliable]
                 if args.exclude_unreliable else rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(plot_rows))
    w = 0.38
    for r, lab, col, off in (
            (3, "lensing-blind (Commander-style)", "#c44e52", -w / 2),
            (4, "lensing-aware (joint sampler)", "#4c72b0", +w / 2)):
        vals = np.array([row[r] / row[2] - 1.0 for row in plot_rows])
        errs = np.array([row[r + 2] / row[2] for row in plot_rows])
        ax.bar(x + off, vals, w, yerr=errs, label=lab, color=col, capsize=3)
    ax.axhline(0.0, color="k", lw=1)
    for i, row in enumerate(plot_rows):
        if (row[0], row[1]) in unreliable:
            ax.axvspan(i - 0.5, i + 0.5, color="0.85", zorder=0)
            ax.text(i, ax.get_ylim()[1] * 0.92, "phi not\nequilibrated",
                    ha="center", va="top", fontsize=7, color="0.35")
    ax.set_xticks(x)
    ax.set_xticklabels([f"[{r[0]},{r[1]})" for r in plot_rows])
    ax.set_xlabel(r"$\ell$ bin")
    ax.set_ylabel(r"fractional bias in $C_\ell^{TT}$" "\n" r"(posterior $-$ correct-model expectation)/expectation")
    sub = (r"$\ell_{\max}$" + f"={args.lmax}, same simulation"
           + (r", bins with unequilibrated $\phi$ omitted"
              if args.exclude_unreliable else ""))
    ax.set_title(r"Lensing-aware joint sampling reduces $C_\ell^{TT}$ bias"
                 "\n" + sub, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=160)
    print(f"Saved {args.out}.png")


if __name__ == "__main__":
    main()
